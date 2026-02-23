# lumina_video_flutter iOS podspec
#
# XCFramework build (required before pod install):
#   ./scripts/build-ios.sh
#   xcodebuild -create-xcframework \
#     -library target/aarch64-apple-ios/release/liblumina_video_ios.a \
#     -headers include/ \
#     -library target/aarch64-apple-ios-sim/release/liblumina_video_ios.a \
#     -headers include/ \
#     -output packages/lumina_video_flutter/ios/Frameworks/LuminaVideo.xcframework
#   cp include/LuminaVideo.h packages/lumina_video_flutter/ios/Classes/CLuminaVideo/include/LuminaVideo.h

Pod::Spec.new do |s|
  s.name             = 'lumina_video_flutter'
  s.version          = '0.1.0'
  s.summary          = 'Hardware-accelerated zero-copy video player for Flutter.'
  s.homepage         = 'https://github.com/lumina-video/lumina-video'
  s.license          = { :type => 'MIT' }
  s.author           = 'lumina-video'
  s.source           = { :path => '.' }

  s.platform         = :ios, '16.0'
  s.swift_version    = '5.9'

  s.source_files     = 'Classes/**/*.swift', 'Classes/CLuminaVideo/include/*.h'
  s.public_header_files = 'Classes/CLuminaVideo/include/LuminaVideo.h'
  s.module_map       = 'Classes/CLuminaVideo/module.modulemap'
  s.preserve_paths   = 'Classes/CLuminaVideo/**'

  s.vendored_frameworks = 'Frameworks/LuminaVideo.xcframework'
  s.frameworks       = 'AVFoundation', 'CoreMedia', 'Metal', 'IOSurface', 'QuartzCore', 'Security'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS' => '-lz -liconv -lbz2 -lc++',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'x86_64',
    'SWIFT_INCLUDE_PATHS' => '$(PODS_TARGET_SRCROOT)/Classes/CLuminaVideo',
  }

  s.dependency 'Flutter'
end
