# IT160IU Data Mining Project

**Student Names:**

Tran Khanh Tai - ITITIU21300

Phan Nguyen Hung Cuong - ITDSIU21078

Nguyen Hoang Anh Thu - ITITIU22241

Pham Nguyen Phan Anh - ITDSIU22147

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

---

## Basic Preprocessing

## Step 1:

Data Analysis

In this section, data analysis are automatically conducted when you load the dataset using the function ReadAndReturnDataLocal(path).
The following are tracked across all attributes: Number of Attributes, Instances, Class Attribute, a summary of all attributes, Min Values, Max Values,
Standard Deviation, Variance, Mean/Mode, Missing Values Count.

Median is currently Work In Progress.

Some are applicable only to Numeric Data.

Afterward, we count the amount of appearances for all instance in all attributes. (EX: Gender have Male: 5022.0 Female: 4978.0)

## Step 2:

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

---

## Step 3: Improved Classification (ImprovedClassifiers)

The `ImprovedClassifiers` class extends `WekaClassifier` to implement advanced ensemble methods that improve prediction accuracy.

### Ensemble Methods Implemented

#### 1. **Bagging** (Bootstrap Aggregating)

- Trains 10 models on bootstrap samples
- Combines predictions via majority voting
- Reduces variance and overfitting

#### 2. **AdaBoost** (Adaptive Boosting)

- Iteratively trains 10 weak learners
- Focuses on misclassified instances
- Assigns weights to each classifier

#### 3. **Voting Ensemble**

- Combines 4 different algorithms:
  - J48 (Decision Tree)
  - NaiveBayes
  - SMO (SVM)
  - RandomForest
- Uses majority voting for final prediction

#### 4. **Stacking** (Meta-Learning)

- **Level 0 (Base Learners):**
  - J48
  - NaiveBayes
  - SMO
- **Level 1 (Meta-Learner):**
  - RandomForest (combines base predictions)

### Performance Metrics

Each ensemble method reports:

- **Accuracy** - Overall classification accuracy
- **Precision** - Weighted precision across classes
- **Recall** - Weighted recall across classes
- **F-Measure** - Harmonic mean of precision and recall
- **Kappa** - Agreement measure
- **ROC Area** - Area under ROC curve
- **Confusion Matrix** - Detailed prediction breakdown
- **Training Time** - Time taken to build the model
- **Evaluation Time** - Time taken for cross-validation

### Why Use Ensemble Methods?

| Method   | Advantage                  | Best For                       |
| -------- | -------------------------- | ------------------------------ |
| Bagging  | Reduces overfitting        | High variance models           |
| AdaBoost | Improves weak learners     | Sequential learning            |
| Voting   | Diverse perspectives       | Combining different algorithms |
| Stacking | Learns optimal combination | Complex datasets               |

### Expected Improvements

Ensemble methods typically achieve:

- 2-5% higher accuracy than base classifiers
- Better generalization to unseen data
- More robust predictions
- Lower variance in results

---

## Running the Complete Pipeline

```java
public static void main(String[] args) throws Exception {
    // Step 1: Load and Preprocess
    Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
    data.setClass(data.attribute("Heart Disease Status"));
    data = DataProcessing.BasicPreprocessing(data);
    data = DataProcessing.OversamplingData(data);

    // Step 2: Baseline Classification
    WekaClassifier j48 = new WekaClassifier("J48");
    j48.loadData(data);
    j48.trainClassifier();
    j48.evaluateModel();

    // Step 3: Improved Classification
    ImprovedClassifiers bagging = new ImprovedClassifiers("BAGGING", "J48");
    bagging.loadData(data);
    bagging.trainClassifier();
    bagging.evaluateModel();
}
```

---

## Models Directory

After running the project, trained models are saved in `models/`:

- `J48_model.model` - Basic J48 classifier
- `BAGGING_model.model` - Bagging ensemble
- `ADABOOST_model.model` - AdaBoost ensemble
- `VOTING_model.model` - Voting ensemble
- `STACKING_model.model` - Stacking ensemble

Load saved models using:

```java
WekaClassifier loaded = new WekaClassifier("J48");
loaded.loadModel("models/J48_model.model");
```

---

## References

- [Weka Documentation](https://www.cs.waikato.ac.nz/ml/weka/)
- [Ensemble Methods Overview](https://scikit-learn.org/stable/modules/ensemble.html)
- [Heart Disease Dataset](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)
