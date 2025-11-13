plugins {
    id("java")
    id("application")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    // Weka library for data mining
    implementation("nz.ac.waikato.cms.weka:weka-stable:3.8.6")
    
    // Apache Commons for data processing
    implementation("org.apache.commons:commons-csv:1.10.0")
    implementation("org.apache.commons:commons-math3:3.6.1")
    
    // Logging
    implementation("org.slf4j:slf4j-simple:2.0.9")
    
    // Testing
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    implementation("nz.ac.waikato.cms.weka:weka-dev:3.9.6")
    implementation("nz.ac.waikato.cms.weka:chiSquaredAttributeEval:1.0.4")
}

application {
    mainClass.set("org.classpj.Main")
}

tasks.test {
    useJUnitPlatform()
}

// Task to run preprocessor
tasks.register<JavaExec>("runPreprocessor") {
    group = "application"
    description = "Run data preprocessing"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.example.DataPreprocessor")
    if (project.hasProperty("args")) {
        args = (project.property("args") as String).split(",")
    }
}

// Task to run classifier
tasks.register<JavaExec>("runClassifier") {
    group = "application"
    description = "Run basic classifier"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.example.WekaClassifier")
    if (project.hasProperty("args")) {
        args = (project.property("args") as String).split(",")
    }
}

// Task to run improved classifier
tasks.register<JavaExec>("runImproved") {
    group = "application"
    description = "Run improved classifier"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.example.ImprovedClassifier")
    if (project.hasProperty("args")) {
        args = (project.property("args") as String).split(",")
    }
}

// Task to run evaluator
tasks.register<JavaExec>("runEvaluator") {
    group = "application"
    description = "Run model evaluator"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.example.ModelEvaluator")
    if (project.hasProperty("args")) {
        args = (project.property("args") as String).split(",")
    }
}