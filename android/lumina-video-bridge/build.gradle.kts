/**
 * lumina-video Android Bridge
 *
 * Provides Java/Kotlin integration for zero-copy video rendering
 * with ExoPlayer and lumina-video's Rust rendering pipeline.
 */

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.luminavideo.bridge"
    compileSdk = 35

    defaultConfig {
        minSdk = 26  // Required for HardwareBuffer API

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    // ExoPlayer / Media3 for video playback
    implementation("androidx.media3:media3-exoplayer:1.2.1")
    implementation("androidx.media3:media3-common:1.2.1")

    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")

    // Lifecycle (for LuminaVideo lifecycle observer)
    implementation("androidx.lifecycle:lifecycle-common:2.7.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}
