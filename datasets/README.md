# Datasets Directory

This directory contains all data files for the Data Mining project.

## Required Files

### Raw Data (You need to download this)
- **File:** `raw_data.csv`
- **Source:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)
- **Description:** Original dataset before preprocessing

### How to Get the Dataset

1. Visit: https://www.kaggle.com/datasets/oktayrdeki/heart-disease
2. Download the dataset
3. Save it in this directory as `raw_data.csv`

## Generated Files

These files will be automatically created by the framework:

- **`processed_data.arff`** - Cleaned and preprocessed data (Step 1)
- **`train.arff`** - Training dataset split
- **`test.arff`** - Testing dataset split

## File Formats

- **CSV** - Comma-separated values (raw data input)
- **ARFF** - Attribute-Relation File Format (Weka format)

## Directory Structure

```
datasets/
├── README.md             # This file
├── raw_data.csv          # Original dataset (you download this)
├── processed_data.arff   # After preprocessing
├── train.arff            # Training set
└── test.arff             # Testing set
```

## Notes

- Make sure you have downloaded the dataset before running Step 1
- The preprocessing step will handle missing values and normalize the data
- All generated files are in ARFF format for compatibility with Weka
