#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use lumina_video_core::linux_video::ZeroCopyGStreamerDecoder;
use lumina_video_core::video::VideoDecoderBackend;

fn serve_fixture(dir: PathBuf) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let dir = dir.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                if reader.read_line(&mut request).is_err() {
                    return;
                }
                let mut line = String::new();
                while reader.read_line(&mut line).map(|n| n > 2).unwrap_or(false) {
                    line.clear();
                }
                let path = request.split_whitespace().nth(1).unwrap_or("/");
                let path = path.split('?').next().unwrap().trim_start_matches('/');
                let response = match std::fs::read(dir.join(path)) {
                    Ok(body) => {
                        let mime = if path.ends_with(".m3u8") {
                            "application/vnd.apple.mpegurl"
                        } else {
                            "video/mp4"
                        };
                        let mut r = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        r.extend(body);
                        r
                    }
                    Err(_) => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_vec()
                    }
                };
                let _ = stream.write_all(&response);
            });
        }
    });
    port
}

#[test]
fn split_rendition_fmp4_hls_with_large_decode_time_keeps_playing() {
    std::env::set_var("EGUI_VID_FAKE_AUDIO", "1");
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/split_av_hls");
    let port = serve_fixture(dir);
    let url = format!("http://127.0.0.1:{port}/live.m3u8");

    let mut decoder = ZeroCopyGStreamerDecoder::open(&url).expect("open decoder");
    decoder.resume().expect("resume");

    let deadline = Instant::now() + Duration::from_secs(15);
    let mut frames = 0;
    while Instant::now() < deadline && frames < 45 {
        match decoder.decode_next() {
            Ok(Some(_)) => frames += 1,
            Ok(None) => {}
            Err(e) => panic!("decode error after {frames} frames: {e}"),
        }
    }
    assert!(frames >= 45, "only {frames} of 60 frames decoded");
}
