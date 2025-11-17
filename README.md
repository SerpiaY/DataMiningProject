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
============================================================================================================
## Basic Preprocessing

### Step 1:
Data Analysis

In this section, data analysis are automatically conducted when you load the dataset using the function ReadAndReturnDataLocal(path).
The following are tracked across all attributes: Number of Attributes, Instances, Class Attribute, a summary of all attributes, Min Values, Max Values,
Standard Deviation, Variance, Mean/Mode, Missing Values Count.

Median is currently Work In Progress.

Some are applicable only to Numeric Data.

Afterward, we count the amount of appearances for all instance in all attributes. (EX: Gender have Male:  5022.0 Female:  4978.0)

### Step 2:
Pre-processing data

After counting all missing values, we use the ReplaceMissingValues Filter to replace all missing values with either mode or mean from the respective class type.

Next, we remove all Duplicates using another filter, and lastly, we apply the Interquartile Range Filter, this filter creates a seperate column for IQR outliers and Extreme Values (in which we found none).

### Attribute Selection
The 2 Pearson and Chi-Square Correlation test are mainly to show which attributes have it the highest correlation to the class attributes. After that, apply the remove function by other code means to remove the specifix attributes lowest on the list

### Oversampling and Resampling within
2 functions exist for oversampling and resampling datasets, since the data is very skewed. Oversampling uses SMOTE and resampling are the Resample Filter. Use either one to check which one produces a better results.

### Train and Test (Stratified Folds)
TrainTestSplits is used to split the train and test sets, this use stratified folds of 10 and return an array of instances ( Instances[] ), the first one is the train set, second one is the test set.

### Additional
Charts

To create a chart, choose either a histogram or a normal bar chart, each one correspond to numeric and nominal, input the dataset and the index of the attribute, and let it go ham.
