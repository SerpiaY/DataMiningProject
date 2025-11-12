# IT160IU Data Mining Project

**Student Names:**

Tran Khanh Tai - ITITIU21300 

Phan Nguyen Hung Cuong - ITDSIU21078

Nguyen Hoang Anh Thu - ITITIU22241

Pham Nguyen Phan Anh  - ITDSIU22147

**Fall 2025**

---

## Project Structure

```
DataMiningProject/
├── src/
│   └── main/
│       └── java/
│           └── org/
│               └── example/
│                   ├── Main.java
│                   ├── DataPreprocessor.java
│                   ├── WekaClassifier.java
│                   ├── ImprovedClassifier.java
│                   └── ModelEvaluator.java
├── datasets/
│   ├── raw_data.csv
│   ├── train.arff
│   └── test.arff
├── models/
│   └── [Generated .model files]
├── build.gradle
├── settings.gradle
├── gradlew
├── gradlew.bat
└── README.md
```

---

## Setup Instructions

### 1. Prerequisites
- Java 11 or higher
- Gradle (wrapper included)

### 2. Download Dataset
- [Heart Disease](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) 

Place the CSV file in `datasets/` folder.

### 3. Build the Project

```bash
# On Linux/Mac
./gradlew build

# On Windows
gradlew.bat build
```

---

## Quick Command Reference

```bash
# Build project
./gradlew build

# Run main menu
./gradlew run

# Individual steps
./gradlew runPreprocessor -Pargs="input.csv,output.arff"
./gradlew runClassifier -Pargs="data.arff"
./gradlew runImproved -Pargs="data.arff"

# Clean build
./gradlew clean build

# Create distribution
./gradlew dist
```