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

The `ModelEvaluators` class provides comprehensive comparison of all classification models using results captured during Steps 2 and 3. **No re-evaluation is performed** - Step 4 generates reports instantly from already-computed metrics.

### Features

#### 1. **Model Comparison Report**

- Side-by-side comparison table of all models
- Displays: Accuracy, Precision, Recall, F-Measure, Kappa, ROC Area, Training Time, Evaluation Time
- Identifies best performers in each category

#### 2. **Experimental Analysis & Remarks**

- Overall best model recommendation based on balanced performance
- Runtime performance analysis with averages
- Ensemble vs Base classifier comparison
- Statistical insights (above/below average performers)

#### 3. **CSV Export**

- Saves all metrics to `evaluation_results.csv`
- Ready for further analysis, visualization, or reporting

### How It Works

Step 4 does **NOT re-evaluate models**. Instead:

1. **During Steps 2 & 3:** Evaluation results and timing are captured when models are trained
2. **Step 4:** Uses the captured `ModelResult` objects to generate comparison reports
3. **Result:** Instant report generation without additional computation

### Usage Example

```java
// During Steps 2 and 3, capture evaluation results with timing
List<ModelResult> allResults = new ArrayList<>();

// Step 2: Train and evaluate base classifiers
for (String classifierType : classifierTypes) {
    WekaClassifier classifier = new WekaClassifier(classifierType);
    classifier.loadData(filePath);

    // Measure training time
    long trainStart = System.nanoTime();
    classifier.trainClassifier();
    long trainEnd = System.nanoTime();
    double trainTime = (trainEnd - trainStart) / 1_000_000_000.0;

    // Measure evaluation time
    long evalStart = System.nanoTime();
    Evaluation eval = classifier.evaluateModel();
    long evalEnd = System.nanoTime();
    double evalTime = (evalEnd - evalStart) / 1_000_000_000.0;

    // Store result
    allResults.add(new ModelResult(classifierType, eval, trainTime, evalTime));
}

// Step 3: Train and evaluate Bagging ensembles (similar timing capture)

// Step 4: Generate comparison reports using captured results
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
J48                  |     77.78% |     0.6853 |     0.7778 |     0.7136 |     0.0070 |     0.5122 |       0.91 |       4.25
NaiveBayes           |     80.00% |     NaN    |     0.8000 |     NaN    |     0.0000 |     0.4931 |       0.07 |       0.44
BAGGING (J48)        |     79.12% |     0.6868 |     0.7912 |     0.7134 |     0.0050 |     0.5014 |       4.29 |      35.24
BAGGING (NaiveBayes) |     80.00% |     NaN    |     0.8000 |     NaN    |     0.0000 |     0.4967 |       0.23 |       2.36

=== BEST PERFORMERS ===
Highest Accuracy:     NaiveBayes (80.00%)
Highest F-Measure:    J48 (0.7136)
Highest ROC Area:     J48 (0.5122)
Fastest Training:     NaiveBayes (0.07s)
Fastest Evaluation:   NaiveBayes (0.44s)

====================================================================================================
EXPERIMENTAL RESULTS ANALYSIS & REMARKS
====================================================================================================

=== OVERALL BEST MODEL ===
Model: BAGGING (J48)
Reasoning: Balanced performance across accuracy, F-Measure, and ROC Area

=== KEY OBSERVATIONS ===

1. Accuracy Analysis:
   - Average Accuracy: 79.2250%
   - BAGGING (J48) performs close to average

2. Runtime Performance:
   - Average Training Time: 1.38s
   - NaiveBayes trains faster than average (0.07s)
   - BAGGING (NaiveBayes) trains faster than average (0.23s)

3. Ensemble Methods vs Base Classifiers:
   - Base Classifiers Average Accuracy: 78.89%
   - Ensemble Methods Average Accuracy: 79.56%
   - Ensemble Improvement: +0.67%
```

### Key Insights from Step 4

1. **Runtime Performance:**

   - NaiveBayes is fastest for training and evaluation
   - Bagging methods require more time but offer better accuracy
   - Trade-off between speed and accuracy must be considered

2. **Ensemble Improvement:**

   - Bagging ensemble typically achieves higher accuracy over base classifiers
   - Improved F-Measure and ROC Area demonstrate better classification
   - Bagging reduces overfitting through model averaging

3. **Important Notes:**
   - **No Re-evaluation:** Step 4 uses results captured during Steps 2 & 3
   - **Efficient Reporting:** Comparison report generated instantly without re-running models
   - **Runtime Tracking:** Training and evaluation times measured during initial model runs
   - **Demo Friendly:** Simplified to 2 base + 2 Bagging models for quick execution

---

## Running the Complete Pipeline

```java
public static void main(String[] args) throws Exception {
    // Step 1: Load and Preprocess
    Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
    data.setClass(data.attribute("Heart Disease Status"));
    data = DataProcessing.BasicPreprocessing(data);
    data = DataProcessing.OversamplingData(data);

    String filePath = "datasets/heart_disease.arff";
    String[] classifierTypes = {"J48", "NaiveBayes"};
    String[] baggingBaseClassifiers = {"J48", "NaiveBayes"};

    // Store results for Step 4 comparison
    List<ModelResult> allResults = new ArrayList<>();

    // Step 2: Train and evaluate base classifiers
    for (String type : classifierTypes) {
        WekaClassifier classifier = new WekaClassifier(type);
        classifier.loadData(filePath);

        long trainStart = System.nanoTime();
        classifier.trainClassifier();
        double trainTime = (System.nanoTime() - trainStart) / 1_000_000_000.0;

        long evalStart = System.nanoTime();
        Evaluation eval = classifier.evaluateModel();
        double evalTime = (System.nanoTime() - evalStart) / 1_000_000_000.0;

        allResults.add(new ModelResult(type, eval, trainTime, evalTime));
        classifier.saveModel("models/" + type + "_model.model");
    }

    // Step 3: Train and evaluate Bagging ensembles
    for (String baseType : baggingBaseClassifiers) {
        ImprovedClassifiers bagging = new ImprovedClassifiers(baseType);
        bagging.loadData(filePath);

        long trainStart = System.nanoTime();
        bagging.trainClassifier();
        double trainTime = (System.nanoTime() - trainStart) / 1_000_000_000.0;

        long evalStart = System.nanoTime();
        Evaluation eval = bagging.evaluateModel();
        double evalTime = (System.nanoTime() - evalStart) / 1_000_000_000.0;

        String modelName = "BAGGING (" + baseType + ")";
        allResults.add(new ModelResult(modelName, eval, trainTime, evalTime));
        bagging.saveModel("models/Bagging_" + baseType + "_model.model");
    }

    // Step 4: Generate comparison reports (no re-evaluation)
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
