package org.classpj.Models;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.Random;

public class WekaClassifier {

    private Classifier classifier;
    private Instances trainData;
    private String modelName;

    /**
     * Constructor - Initialize with a specific classifier
     * @param classifierType: "J48", "NaiveBayes", "SMO", "RandomForest"
     */
    public WekaClassifier(String classifierType) {
        this.modelName = classifierType;

        switch(classifierType.toUpperCase()) {
            case "J48":
                // Decision Tree (C4.5)
                classifier = new J48();
                try {
                    String[] options = new String[]{"-C", "0.25", "-M", "2"};
                    ((J48)classifier).setOptions(options);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;

            case "NAIVEBAYES":
                // Naive Bayes
                classifier = new NaiveBayes();
                break;

            case "SMO":
                // Support Vector Machine
                classifier = new SMO();
                break;

            case "RANDOMFOREST":
                // Random Forest
                classifier = new RandomForest();
                try {
                    ((RandomForest)classifier).setNumIterations(100);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;

            default:
                System.out.println("Unknown classifier type. Using J48 as default.");
                classifier = new J48();
                this.modelName = "J48";
        }
    }

    /**
     * Load data from ARFF file
     */
    public void loadData(String arffFilePath) {
        try {
            DataSource source = new DataSource(arffFilePath);
            trainData = source.getDataSet();

            // Set class attribute (last attribute by default)
            if (trainData.classIndex() == -1) {
                trainData.setClassIndex(trainData.numAttributes() - 1);
            }

            System.out.println("Data loaded successfully!");
            System.out.println("Number of instances: " + trainData.numInstances());
            System.out.println("Number of attributes: " + trainData.numAttributes());
            System.out.println("Class attribute: " + trainData.classAttribute().name());

        } catch (Exception e) {
            System.err.println("Error loading data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Train the classifier
     */
    public void trainClassifier() {
        try {
            System.out.println("\n=== Training " + modelName + " Classifier ===");
            long startTime = System.currentTimeMillis();

            classifier.buildClassifier(trainData);

            long endTime = System.currentTimeMillis();
            double trainingTime = (endTime - startTime) / 1000.0;

            System.out.println("Training completed in " + trainingTime + " seconds");

        } catch (Exception e) {
            System.err.println("Error training classifier: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Evaluate using 10-fold cross-validation
     */
    public Evaluation evaluateModel() {
        Evaluation eval = null;
        try {
            System.out.println("\n=== 10-Fold Cross-Validation ===");

            eval = new Evaluation(trainData);
            long startTime = System.currentTimeMillis();

            // Perform 10-fold cross-validation
            eval.crossValidateModel(classifier, trainData, 10, new Random(1));

            long endTime = System.currentTimeMillis();
            double evalTime = (endTime - startTime) / 1000.0;

            // Print evaluation results
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println("=== Detailed Accuracy By Class ===");
            System.out.println(eval.toClassDetailsString());
            System.out.println("=== Confusion Matrix ===");
            System.out.println(eval.toMatrixString());

            // Additional statistics
            System.out.println("\n=== Performance Metrics ===");
            System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
            System.out.println("Kappa statistic: " + String.format("%.4f", eval.kappa()));
            System.out.println("Mean absolute error: " + String.format("%.4f", eval.meanAbsoluteError()));
            System.out.println("Root mean squared error: " + String.format("%.4f", eval.rootMeanSquaredError()));
            System.out.println("Evaluation time: " + evalTime + " seconds");

        } catch (Exception e) {
            System.err.println("Error evaluating model: " + e.getMessage());
            e.printStackTrace();
        }
        return eval;
    }

    /**
     * Save the trained model to a file
     */
    public void saveModel(String outputPath) {
        try {
            // Create directory if it doesn't exist
            File file = new File(outputPath);
            file.getParentFile().mkdirs();

            SerializationHelper.write(outputPath, classifier);
            System.out.println("\nModel saved to: " + outputPath);

        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Load a saved model
     */
    public void loadModel(String modelPath) {
        try {
            classifier = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Model loaded from: " + modelPath);

        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Make prediction on a single instance
     */
    public double classifyInstance(Instances unlabeledData, int instanceIndex) {
        try {
            return classifier.classifyInstance(unlabeledData.instance(instanceIndex));
        } catch (Exception e) {
            System.err.println("Error classifying instance: " + e.getMessage());
            e.printStackTrace();
            return -1;
        }
    }

    /**
     * Get distribution for an instance (probabilities for each class)
     */
    public double[] getDistribution(Instances data, int instanceIndex) {
        try {
            return classifier.distributionForInstance(data.instance(instanceIndex));
        } catch (Exception e) {
            System.err.println("Error getting distribution: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Test on separate test set
     */
    public void evaluateOnTestSet(String testSetPath) {
        try {
            System.out.println("\n=== Evaluating on Test Set ===");

            // Load test data
            DataSource source = new DataSource(testSetPath);
            Instances testData = source.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            // Evaluate
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(classifier, testData);

            System.out.println(eval.toSummaryString("\n=== Test Set Results ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            System.err.println("Error evaluating on test set: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Getters
    public Classifier getClassifier() {
        return classifier;
    }

    public Instances getTrainData() {
        return trainData;
    }

    public String getModelName() {
        return modelName;
    }
}