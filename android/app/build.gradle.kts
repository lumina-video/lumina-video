/**
 * lumina-video Demo Application
 *
 * Test application for demonstrating lumina-video zero-copy video rendering
 * with ExoPlayer on Android using GameActivity for egui/winit integration.
 */

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.luminavideo.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.luminavideo.demo"
        minSdk = 26      // Required for HardwareBuffer API
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // NDK/CMake configuration for native library
        ndk {
            // Supported ABIs for lumina-video
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // NDK configuration
    ndkVersion = "26.1.10909125"

    // CMake configuration for building the native library
    // Note: If you're building liblumina_video.so separately with cargo-ndk,
    // you can comment this out and place the .so files in jniLibs/
    /*
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    */

    // Source sets for native libraries
    sourceSets {
        getByName("main") {
            // Native libraries built by cargo-ndk go here
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    buildFeatures {
        prefab = true
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // lumina-video bridge module
    implementation(project(":lumina-video-bridge"))

    // Media3 / ExoPlayer
    implementation("androidx.media3:media3-exoplayer:1.9.0")
    implementation("androidx.media3:media3-exoplayer-hls:1.9.0")
    implementation("androidx.media3:media3-exoplayer-dash:1.9.0")
    implementation("androidx.media3:media3-common:1.9.0")
    implementation("androidx.media3:media3-ui:1.9.0")

    // AndroidX GameActivity for egui/winit android-game-activity feature
    implementation("androidx.games:games-activity:4.0.0")

    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
