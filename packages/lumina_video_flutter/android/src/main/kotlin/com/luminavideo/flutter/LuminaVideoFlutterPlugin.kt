package com.luminavideo.flutter

import android.app.Activity
import android.os.Handler
import android.os.Looper
import android.view.Surface
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.video.VideoFrameMetadataListener
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry
import java.util.concurrent.atomic.AtomicInteger

private class PlayerEntry(
    val player: ExoPlayer,
    val textureEntry: TextureRegistry.SurfaceTextureEntry,
    val surface: Surface,
    var playerListener: Player.Listener,
    val positionUpdater: Runnable,
    val eventChannel: EventChannel,
    var eventSink: EventChannel.EventSink? = null,
    var videoWidth: Int = 0,
    var videoHeight: Int = 0,
    var wasPlayingBeforeBackground: Boolean = false,
    // Diagnostics
    var frameCount: Int = 0,
    var fpsWindowStart: Long = System.nanoTime(),
    var currentFps: Double = 0.0,
    var sourceUrl: String = "",
)

class LuminaVideoFlutterPlugin : FlutterPlugin, MethodChannel.MethodCallHandler, ActivityAware {

    companion object {
        // NOTE: Global mutable state — intentional rule exception.
        // Must survive plugin instance recreation across hot restarts. If this
        // were instance state, hot restart would reset it to 1, colliding with
        // stale native player IDs still alive in the process. AtomicInteger is
        // defensive (main-thread-only access, but zero cost).
        private val nextPlayerId = AtomicInteger(1)
    }

    private lateinit var channel: MethodChannel
    private lateinit var textureRegistry: TextureRegistry
    private lateinit var messenger: io.flutter.plugin.common.BinaryMessenger
    private val players = mutableMapOf<Int, PlayerEntry>()
    private val handler = Handler(Looper.getMainLooper())

    private var activity: Activity? = null
    private var lifecycleObserver: DefaultLifecycleObserver? = null

    // MARK: - FlutterPlugin

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        textureRegistry = binding.textureRegistry
        messenger = binding.binaryMessenger
        channel = MethodChannel(binding.binaryMessenger, "lumina_video_flutter")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        unregisterLifecycleObserver()
        for (id in players.keys.toList()) { destroyPlayer(id) }
    }

    // MARK: - MethodCallHandler

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "create" -> handleCreate(call, result)
            "play" -> handlePlayerCommand(call, result) { it.player.play() }
            "pause" -> handlePlayerCommand(call, result) { it.player.pause() }
            "seek" -> {
                val position = call.argument<Double>("position")
                if (position == null) { result.error("INVALID_ARGS", "Missing position", null); return }
                handlePlayerCommand(call, result) { it.player.seekTo((position * 1000).toLong()) }
            }
            "setMuted" -> {
                val muted = call.argument<Boolean>("muted")
                if (muted == null) { result.error("INVALID_ARGS", "Missing muted", null); return }
                handlePlayerCommand(call, result) { it.player.volume = if (muted) 0f else 1f }
            }
            "setVolume" -> {
                val volume = call.argument<Int>("volume")
                if (volume == null) { result.error("INVALID_ARGS", "Missing volume", null); return }
                handlePlayerCommand(call, result) { it.player.volume = (volume / 100f).coerceIn(0f, 1f) }
            }
            "destroy" -> handleDestroy(call, result)
            "_debugGetPlayersLive" -> {
                if (BuildConfig.DEBUG) {
                    result.success(players.size)
                } else {
                    result.notImplemented()
                }
            }
            else -> result.notImplemented()
        }
    }

    // MARK: - Create

    private fun handleCreate(call: MethodCall, result: MethodChannel.Result) {
        val url = call.argument<String>("url")
        if (url == null) {
            result.error("INVALID_ARGS", "Missing url", null)
            return
        }

        val context = activity ?: run {
            result.error("NO_ACTIVITY", "No activity attached", null)
            return
        }

        val playerId = nextPlayerId.getAndIncrement()

        val textureEntry = textureRegistry.createSurfaceTexture()
        val surfaceTexture = textureEntry.surfaceTexture()
        val surface = Surface(surfaceTexture)

        val exoPlayer = ExoPlayer.Builder(context).build()

        // Position updater — runs every 250ms on main handler
        val positionUpdater = object : Runnable {
            override fun run() {
                val entry = players[playerId] ?: return
                pushEvent(entry)
                handler.postDelayed(this, 250)
            }
        }

        // EventChannel for this player
        val channelName = "lumina_video_flutter/events/$playerId"
        val eventChannel = EventChannel(messenger, channelName)

        // Single entry object — all closures capture this reference
        val entry = PlayerEntry(
            player = exoPlayer,
            textureEntry = textureEntry,
            surface = surface,
            playerListener = object : Player.Listener {},  // replaced below
            positionUpdater = positionUpdater,
            eventChannel = eventChannel,
            sourceUrl = url,
        )
        players[playerId] = entry

        // Set up stream handler with onListen snapshot
        eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                entry.eventSink = events
                events?.success(buildEventMap(entry))
            }

            override fun onCancel(arguments: Any?) {
                entry.eventSink = null
            }
        })

        // Player listener
        val listener = object : Player.Listener {
            override fun onPlaybackStateChanged(playbackState: Int) {
                pushEvent(entry)
            }

            override fun onIsPlayingChanged(isPlaying: Boolean) {
                pushEvent(entry)
            }

            override fun onVideoSizeChanged(videoSize: VideoSize) {
                if (videoSize.width > 0 && videoSize.height > 0) {
                    surfaceTexture.setDefaultBufferSize(videoSize.width, videoSize.height)
                    entry.videoWidth = videoSize.width
                    entry.videoHeight = videoSize.height
                    pushEvent(entry)
                }
            }

            override fun onPlayerError(error: PlaybackException) {
                pushEvent(entry, error = mapOf(
                    "code" to "EXOPLAYER_ERROR",
                    "message" to (error.message ?: "Unknown"),
                    "platformCode" to error.errorCode,
                ))
            }
        }
        entry.playerListener = listener
        exoPlayer.addListener(listener)
        exoPlayer.setVideoSurface(surface)

        // Per-frame callback for FPS measurement
        exoPlayer.setVideoFrameMetadataListener { _, _, _, _ ->
            entry.frameCount++
            val now = System.nanoTime()
            val elapsed = (now - entry.fpsWindowStart) / 1_000_000_000.0
            if (elapsed >= 1.0) {
                entry.currentFps = entry.frameCount / elapsed
                entry.frameCount = 0
                entry.fpsWindowStart = now
            }
        }

        exoPlayer.setMediaItem(MediaItem.fromUri(url))
        exoPlayer.prepare()

        // Start position timer
        handler.postDelayed(positionUpdater, 250)

        // Return initial state
        val response = buildEventMap(entry).toMutableMap()
        response["playerId"] = playerId
        response["textureId"] = textureEntry.id()
        result.success(response)
    }

    // MARK: - Player commands

    private fun handlePlayerCommand(
        call: MethodCall,
        result: MethodChannel.Result,
        action: (PlayerEntry) -> Unit
    ) {
        val playerId = call.argument<Int>("playerId")
        if (playerId == null) {
            result.error("INVALID_ARGS", "Missing playerId", null)
            return
        }
        val entry = players[playerId]
        if (entry == null) {
            result.error("UNKNOWN_PLAYER", "Player not found", null)
            return
        }
        action(entry)
        result.success(null)
    }

    // MARK: - Destroy

    private fun handleDestroy(call: MethodCall, result: MethodChannel.Result) {
        val playerId = call.argument<Int>("playerId")
        if (playerId == null) {
            result.error("INVALID_ARGS", "Missing playerId", null)
            return
        }
        if (players[playerId] == null) {
            result.error("UNKNOWN_PLAYER", "Player not found", null)
            return
        }
        destroyPlayer(playerId)
        result.success(null)
    }

    private fun destroyPlayer(id: Int) {
        val entry = players.remove(id) ?: return
        // 1. Remove listener — no more callbacks
        entry.player.removeListener(entry.playerListener)
        handler.removeCallbacks(entry.positionUpdater)
        // 2. Tear down EventChannel
        entry.eventSink = null
        entry.eventChannel.setStreamHandler(null)
        // 3. Release ExoPlayer + surfaces
        entry.player.setVideoSurface(null)
        entry.player.release()
        entry.surface.release()
        entry.textureEntry.release()
    }

    // MARK: - Event building (centralized)

    private fun mapExoState(playbackState: Int, playWhenReady: Boolean): Int {
        return when (playbackState) {
            Player.STATE_IDLE -> 0      // loading
            Player.STATE_BUFFERING -> 0 // loading
            Player.STATE_READY -> if (playWhenReady) 2 else 3 // playing / paused
            Player.STATE_ENDED -> 4     // ended
            else -> 5                   // error
        }
    }

    private fun buildEventMap(entry: PlayerEntry, error: Map<String, Any?>? = null): Map<String, Any?> {
        val rawDuration = entry.player.duration
        val duration = if (rawDuration == C.TIME_UNSET) -1.0 else rawDuration / 1000.0
        val state = if (error != null) 5 else mapExoState(entry.player.playbackState, entry.player.playWhenReady)
        val map = mutableMapOf<String, Any?>(
            "v" to 1,
            "state" to state,
            "position" to (entry.player.currentPosition / 1000.0),
            "duration" to duration,
        )
        if (entry.videoWidth > 0) {
            map["videoWidth"] = entry.videoWidth
            map["videoHeight"] = entry.videoHeight
        }
        if (error != null) map["error"] = error

        // Diagnostics
        if (entry.currentFps > 0) {
            map["fps"] = entry.currentFps
        }
        val videoFrameRate = entry.player.videoFormat?.frameRate
        if (videoFrameRate != null && videoFrameRate > 0) {
            map["maxFps"] = videoFrameRate.toInt()
        }
        map["zeroCopy"] = false // Android uses SurfaceTexture (GPU copy), not zero-copy IOSurface
        val videoFormat = entry.player.videoFormat
        if (videoFormat != null) {
            map["videoCodec"] = mapMimeToCodecName(videoFormat.sampleMimeType)
        }
        val audioFormat = entry.player.audioFormat
        if (audioFormat != null) {
            map["audioCodec"] = mapMimeToCodecName(audioFormat.sampleMimeType)
        }
        val ext = android.net.Uri.parse(entry.sourceUrl).lastPathSegment
            ?.substringAfterLast('.', "")?.uppercase()
        if (!ext.isNullOrEmpty()) {
            map["format"] = ext
        }

        return map
    }

    private fun mapMimeToCodecName(mime: String?): String {
        return when (mime) {
            "video/avc" -> "H.264"
            "video/hevc" -> "H.265"
            "video/x-vnd.on2.vp8" -> "VP8"
            "video/x-vnd.on2.vp9" -> "VP9"
            "video/av01" -> "AV1"
            "audio/mp4a-latm" -> "AAC"
            "audio/opus" -> "Opus"
            "audio/mpeg" -> "MP3"
            "audio/vorbis" -> "Vorbis"
            "audio/ac3" -> "AC-3"
            "audio/eac3" -> "E-AC-3"
            else -> mime ?: "Unknown"
        }
    }

    private fun pushEvent(entry: PlayerEntry, error: Map<String, Any?>? = null) {
        entry.eventSink?.success(buildEventMap(entry, error))
    }

    // MARK: - ActivityAware

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity
        registerLifecycleObserver()
    }

    override fun onDetachedFromActivityForConfigChanges() {
        unregisterLifecycleObserver()
        activity = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        activity = binding.activity
        registerLifecycleObserver()
    }

    override fun onDetachedFromActivity() {
        unregisterLifecycleObserver()
        activity = null
    }

    // MARK: - Lifecycle observer

    private fun registerLifecycleObserver() {
        if (lifecycleObserver != null) return
        val observer = object : DefaultLifecycleObserver {
            override fun onStop(owner: LifecycleOwner) {
                for ((_, entry) in players) {
                    if (entry.player.isPlaying) {
                        entry.player.pause()
                        entry.wasPlayingBeforeBackground = true
                    }
                }
            }

            override fun onStart(owner: LifecycleOwner) {
                for ((_, entry) in players) {
                    if (entry.wasPlayingBeforeBackground) {
                        entry.player.play()
                        entry.wasPlayingBeforeBackground = false
                    }
                }
            }
        }
        lifecycleObserver = observer
        ProcessLifecycleOwner.get().lifecycle.addObserver(observer)
    }

    private fun unregisterLifecycleObserver() {
        lifecycleObserver?.let { ProcessLifecycleOwner.get().lifecycle.removeObserver(it) }
        lifecycleObserver = null
    }
}
