/**
 * lumina-video Android Project
 *
 * Root settings for the lumina-video Android project including:
 * - :app - Demo application for testing lumina-video
 * - :lumina-video-bridge - ExoPlayer bridge library
 */

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "lumina-video"
include(":app")
include(":lumina-video-bridge")
