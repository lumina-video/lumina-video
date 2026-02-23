import 'package:flutter/material.dart';
import 'package:lumina_video_flutter/lumina_video_flutter.dart';

void main() => runApp(const LuminaVideoExampleApp());

class LuminaVideoExampleApp extends StatelessWidget {
  const LuminaVideoExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lumina Video Example',
      theme: ThemeData.dark(useMaterial3: true),
      home: const PlayerPage(),
    );
  }
}

class PlayerPage extends StatefulWidget {
  const PlayerPage({super.key});

  @override
  State<PlayerPage> createState() => _PlayerPageState();
}

class _PlayerPageState extends State<PlayerPage> {
  late final _player = LuminaPlayer();

  static const _testUrl =
      'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';

  @override
  void initState() {
    super.initState();
    _player.open(_testUrl).then((_) => _player.play()).catchError((Object e) {
      debugPrint('Player init failed: $e');
    });
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Lumina Video')),
      body: ValueListenableBuilder<LuminaPlayerValue>(
        valueListenable: _player,
        builder: (_, val, __) {
          if (val.error != null) {
            return Center(child: Text('Error: ${val.error!.message}'));
          }
          if (!val.isInitialized) {
            return const Center(child: CircularProgressIndicator());
          }
          return Column(
            children: [
              Expanded(
                child: Center(
                  child: AspectRatio(
                    aspectRatio: val.videoSize != null
                        ? val.videoSize!.width / val.videoSize!.height
                        : 16 / 9,
                    child: Texture(textureId: val.textureId),
                  ),
                ),
              ),
              _Controls(player: _player, value: val),
            ],
          );
        },
      ),
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({required this.player, required this.value});

  final LuminaPlayer player;
  final LuminaPlayerValue value;

  String _formatTime(double seconds) {
    if (seconds < 0) return '--:--';
    final m = seconds ~/ 60;
    final s = (seconds % 60).toInt();
    return '$m:${s.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (value.duration > 0)
            Slider(
              value: value.position.clamp(0.0, value.duration),
              max: value.duration,
              onChanged: (v) => player.seek(v),
            ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(_formatTime(value.position)),
              const SizedBox(width: 8),
              Text('/ ${_formatTime(value.duration)}'),
              const Spacer(),
              IconButton(
                icon: Icon(
                  value.state == LuminaPlaybackState.playing
                      ? Icons.pause
                      : Icons.play_arrow,
                ),
                onPressed: () {
                  if (value.state == LuminaPlaybackState.playing) {
                    player.pause();
                  } else {
                    player.play();
                  }
                },
              ),
            ],
          ),
        ],
      ),
    );
  }
}
