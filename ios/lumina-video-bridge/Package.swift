// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LuminaVideoBridge",
    platforms: [.iOS(.v16)],
    products: [
        .library(name: "LuminaVideoBridge", targets: ["LuminaVideoBridge"]),
    ],
    targets: [
        .target(
            name: "CLuminaVideo",
            path: "CHeaders",
            publicHeadersPath: "include"
        ),
        .target(
            name: "LuminaVideoBridge",
            dependencies: ["CLuminaVideo"],
            path: "Sources/LuminaVideoBridge",
            linkerSettings: [
                .linkedLibrary("lumina_video_ios"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Metal"),
                .linkedFramework("IOSurface"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("Security"),
                .linkedFramework("VideoToolbox"),
            ]
        ),
    ]
)
