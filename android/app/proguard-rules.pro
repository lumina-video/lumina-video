# ProGuard rules for lumina-video demo application
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Keep native methods (JNI callbacks from Rust)
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep ExoPlayerBridge for JNI access
-keep class com.luminavideo.bridge.ExoPlayerBridge {
    *;
}

# Keep MainActivity methods called from native code
-keep class com.luminavideo.demo.MainActivity {
    com.luminavideo.bridge.ExoPlayerBridge getBridge();
    androidx.media3.exoplayer.ExoPlayer getPlayer();
    void playVideo(java.lang.String);
    void pauseVideo();
    void resumeVideo();
    void seekTo(long);
    long getCurrentPosition();
    long getDuration();
    boolean isPlaying();
}

# Keep Media3/ExoPlayer classes
-keep class androidx.media3.** { *; }
-dontwarn androidx.media3.**

# Keep GameActivity
-keep class com.google.androidgamesdk.GameActivity { *; }

# Keep HardwareBuffer for zero-copy path
-keep class android.hardware.HardwareBuffer { *; }
