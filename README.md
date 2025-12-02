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

The `ImprovedClassifiers` class extends `WekaClassifier` to implement the Bagging ensemble method for improved prediction accuracy.

### Ensemble Method: Bagging (Bootstrap Aggregating)

The Bagging ensemble method improves classification performance by:

- **Training 10 models** on bootstrap samples (random sampling with replacement)
- **Combining predictions** via majority voting
- **Reducing variance** and preventing overfitting
- **Works with any base classifier** (J48, NaiveBayes, etc.)

### Implementation Details

```java
ImprovedClassifiers bagging = new ImprovedClassifiers("J48");
bagging.loadData(data);
bagging.trainClassifier();
bagging.evaluateModel();
```

### Performance Metrics

Each ensemble evaluation reports:

- **Accuracy** - Overall classification accuracy
- **Precision** - Weighted precision across classes
- **Recall** - Weighted recall across classes
- **F-Measure** - Harmonic mean of precision and recall
- **Kappa** - Agreement measure
- **ROC Area** - Area under ROC curve
- **Confusion Matrix** - Detailed prediction breakdown
- **Training Time** - Time taken to build the model
- **Evaluation Time** - Time taken for 10-fold cross-validation

### Why Use Bagging?

| Advantage                | Benefit                         |
| ------------------------ | ------------------------------- |
| Reduces overfitting      | Better generalization           |
| Stable predictions       | Lower variance in results       |
| Works with any algorithm | Can improve any base classifier |
| Fast training            | Suitable for demo constraints   |

### Expected Improvements

Bagging typically achieves:

- **2-7% higher accuracy** than base classifiers
- **Better F-Measure** and ROC Area scores
- **More robust predictions** on unseen data
- **Lower variance** in cross-validation results

---

## Step 4: Model Evaluation and Comparison Report (ModelEvaluators)

The `ModelEvaluators` class provides comprehensive evaluation and comparison of all classification models, including runtime performance analysis.

### Features

#### 1. **Individual Model Evaluation**

- Evaluates each classifier using 10-fold cross-validation
- Tracks training and evaluation time
- Reports detailed metrics:
  - Accuracy, Precision, Recall, F-Measure
  - Kappa statistic, ROC Area
  - Confusion matrix
  - Per-class statistics

#### 2. **Comparative Analysis**

- Side-by-side comparison table of all models
- Identifies best performers for:
  - Highest Accuracy
  - Highest F-Measure
  - Highest ROC Area
  - Fastest Training
  - Fastest Evaluation

#### 3. **Experimental Remarks**

- Overall best model recommendation
- Runtime performance analysis
- Ensemble vs Base classifier comparison
- Statistical insights (averages, above/below average)

#### 4. **Export Results**

- Saves all metrics to CSV file (`evaluation_results.csv`)
- Suitable for further analysis or reporting

### Usage Example

```java
List<ModelResult> allResults = new ArrayList<>();

// Evaluate base classifiers
WekaClassifier j48 = new WekaClassifier("J48");
ModelResult j48Result = ModelEvaluators.evaluateClassifier(
    j48.getClassifier(), data, "J48");
allResults.add(j48Result);

// Evaluate ensemble methods
ImprovedClassifiers bagging = new ImprovedClassifiers("J48");
ModelResult baggingResult = ModelEvaluators.evaluateClassifier(
    bagging.getEnsembleClassifier(), data, "BAGGING (J48)");
allResults.add(baggingResult);

// Generate comprehensive reports
ModelEvaluators.compareModels(allResults);
ModelEvaluators.generateAnalysisReport(allResults);
ModelEvaluators.saveResultsToCSV(allResults, "evaluation_results.csv");
```

### Evaluation Metrics Tracked

| Metric          | Description                                  |
| --------------- | -------------------------------------------- |
| Accuracy        | Percentage of correct predictions            |
| Precision       | Weighted average of per-class precision      |
| Recall          | Weighted average of per-class recall         |
| F-Measure       | Harmonic mean of precision and recall        |
| Kappa           | Agreement measure (accounting for chance)    |
| ROC Area        | Area under Receiver Operating Characteristic |
| Training Time   | Time to build the model (seconds)            |
| Evaluation Time | Time for 10-fold cross-validation (seconds)  |

### Sample Output

```
====================================================================================================
MODEL COMPARISON REPORT
====================================================================================================

Model                | Accuracy   | Precision  | Recall     | F-Measure  | Kappa      | ROC Area   | Train(s)   | Eval(s)
------------------------------------------------------------------------------------------------------------------------------------
J48                  |     62.44% |     0.6235 |     0.6244 |     0.6239 |     0.2312 |     0.6213 |       1.46 |      12.44
NaiveBayes           |     62.94% |     0.6250 |     0.6294 |     0.6089 |     0.2054 |     0.6552 |       0.05 |       0.49
BAGGING (J48)        |     69.24% |     0.6897 |     0.6924 |     0.6889 |     0.3618 |     0.7339 |      12.83 |     109.82
BAGGING (NaiveBayes) |     62.79% |     0.6232 |     0.6279 |     0.6072 |     0.2021 |     0.6551 |       0.33 |       3.55

=== BEST PERFORMERS ===
Highest Accuracy:     BAGGING (J48) (69.2357%)
Highest F-Measure:    BAGGING (J48) (0.6889)
Highest ROC Area:     BAGGING (J48) (0.7339)
Fastest Training:     NaiveBayes (0.051s)
Fastest Evaluation:   NaiveBayes (0.487s)
```

### Key Insights from Step 4

1. **Runtime Performance:**

   - NaiveBayes is fastest for training and evaluation
   - Bagging methods require more time but offer better accuracy
   - Trade-off between speed and accuracy must be considered

2. **Ensemble Improvement:**

   - Bagging (J48) achieves **+6.8% accuracy** over base J48
   - Improved F-Measure and ROC Area demonstrate better classification
   - Bagging reduces overfitting through model averaging

3. **Recommendations:**
   - **Production Use:** BAGGING (J48) - Best accuracy and performance
   - **Real-Time Applications:** NaiveBayes - Fastest total time (0.538s)
   - **Demo Constraints:** Simplified to 2 base + 2 Bagging models (~3 min runtime)

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
    String[] classifierTypes = {"J48", "NaiveBayes"};
    for (String type : classifierTypes) {
        WekaClassifier classifier = new WekaClassifier(type);
        classifier.loadData(data);
        classifier.trainClassifier();
        classifier.evaluateModel();
        classifier.saveModel("models/" + type + "_model.model");
    }

    // Step 3: Improved Classification with Bagging
    String[] baggingBaseClassifiers = {"J48", "NaiveBayes"};
    for (String baseType : baggingBaseClassifiers) {
        ImprovedClassifiers bagging = new ImprovedClassifiers(baseType);
        bagging.loadData(data);
        bagging.trainClassifier();
        bagging.evaluateModel();
        bagging.saveModel("models/Bagging_" + baseType + "_model.model");
    }

    // Step 4: Comprehensive Evaluation and Comparison
    List<ModelResult> allResults = new ArrayList<>();

    // Evaluate all models
    for (String type : classifierTypes) {
        WekaClassifier classifier = new WekaClassifier(type);
        ModelResult result = ModelEvaluators.evaluateClassifier(
            classifier.getClassifier(), data, type);
        allResults.add(result);
    }

    for (String baseType : baggingBaseClassifiers) {
        ImprovedClassifiers bagging = new ImprovedClassifiers(baseType);
        ModelResult result = ModelEvaluators.evaluateClassifier(
            bagging.getEnsembleClassifier(), data, "BAGGING (" + baseType + ")");
        allResults.add(result);
    }

    // Generate reports
    ModelEvaluators.compareModels(allResults);
    ModelEvaluators.generateAnalysisReport(allResults);
    ModelEvaluators.saveResultsToCSV(allResults, "evaluation_results.csv");
}
```

---

## Output Files

After running the project, the following files are generated:

### Models Directory (`models/`)

- `J48_model.model` - Basic J48 classifier
- `NaiveBayes_model.model` - Basic NaiveBayes classifier
- `Bagging_J48_model.model` - Bagging ensemble with J48
- `Bagging_NaiveBayes_model.model` - Bagging ensemble with NaiveBayes

### Evaluation Results

- `evaluation_results.csv` - Comprehensive metrics for all models

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
