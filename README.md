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
│               └── classpj/
│                   ├── Main.java
│                   ├── Models/
│                      ├── WekaClassifier.java
│                      └──ImprovedClassifier.java
│                   ├── Processors/
│                      ├── DataPreprocessor.java
│                      └── ModelEvaluator.java
│       └── resources/
|           ├── datasets/
│               ├── heart_disease.csv
│               ├── airplane.arff
│               └── credits-g.arff
|           ├── models/
│               └── [Generated .model files]
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

## Pre-processor usage
To use the Preprocessing class, first put the data inside the resources folder, then run the load data function (examples in Main.java), after that, 
there are 2 attribute selectors you can use to determine the best attributes, Pearson Correlation and Chi-Square-Test.
### Note: For Pearson, all data are turned into numeric, and for Chi-Square, all data are nominal.
