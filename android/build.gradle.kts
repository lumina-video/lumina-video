/**
 * lumina-video Android Root Build Configuration
 *
 * This is the root build file for the lumina-video Android project.
 * Common configuration is defined here for all modules.
 */

plugins {
    id("com.android.application") version "8.7.0" apply false
    id("com.android.library") version "8.7.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}

// Clean task to remove build directories
tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
