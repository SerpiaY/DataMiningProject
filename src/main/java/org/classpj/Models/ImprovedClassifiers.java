package org.classpj.Models;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.File;
import java.util.Random;

/**
 * Step 3: Improved Classification (20 pts)
 * Implements Bagging (Bootstrap Aggregating) ensemble method to improve
 * prediction accuracy.
 * 
 * Bagging works by:
 * - Creating multiple bootstrap samples from training data
 * - Training a base classifier on each sample
 * - Combining predictions through majority voting
 * 
 * This reduces variance and improves model stability.
 */
public class ImprovedClassifiers extends WekaClassifier {

    private Bagging baggingClassifier;
    private Instances trainData;

    /**
     * Constructor - Initialize Bagging ensemble
     * 
     * @param baseClassifierType: "J48", "NaiveBayes", "SMO", or "RandomForest"
     */
    public ImprovedClassifiers(String baseClassifierType) {
        super(baseClassifierType);
        initializeBagging(baseClassifierType);
    }

    /**
     * Initialize the Bagging classifier
     */
    private void initializeBagging(String baseClassifierType) {
        try {
            Classifier baseClassifier = createBaseClassifier(baseClassifierType);

            baggingClassifier = new Bagging();
            baggingClassifier.setClassifier(baseClassifier);
            baggingClassifier.setNumIterations(10); // 10 models in ensemble
            baggingClassifier.setBagSizePercent(100); // 100% bootstrap sample size

            System.out.println("\n=== Bagging Ensemble Initialized ===");
            System.out.println("Base Classifier: " + baseClassifierType);
            System.out.println("Number of Models: 10");
            System.out.println("Bootstrap Sample Size: 100%");

        } catch (Exception e) {
            System.err.println("Error initializing Bagging: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Create base classifier instance
     */
    private Classifier createBaseClassifier(String type) throws Exception {
        switch (type.toUpperCase()) {
            case "J48":
                J48 j48 = new J48();
                j48.setOptions(new String[] { "-C", "0.25", "-M", "2" });
                return j48;
            case "NAIVEBAYES":
                return new NaiveBayes();
            default:
                return new J48();
        }
    }

    /**
     * Load data from ARFF file
     */
    @Override
    public void loadData(String arffFilePath) {
        super.loadData(arffFilePath);
        this.trainData = super.getTrainData();
    }

    /**
     * Load Instances directly (for preprocessed data)
     */
    public void loadData(Instances data) {
        this.trainData = new Instances(data);
        if (trainData.classIndex() == -1) {
            trainData.setClassIndex(trainData.numAttributes() - 1);
        }
        System.out.println("\n=== Data Loaded for Bagging Ensemble ===");
        System.out.println("Instances: " + trainData.numInstances());
        System.out.println("Attributes: " + trainData.numAttributes());
        System.out.println("Class: " + trainData.classAttribute().name());
    }

    /**
     * Train the Bagging ensemble
     */
    @Override
    public void trainClassifier() {
        try {
            if (trainData == null) {
                System.err.println("No training data loaded!");
                return;
            }

            System.out.println("\n=== Training Bagging Ensemble ===");
            long startTime = System.currentTimeMillis();

            baggingClassifier.buildClassifier(trainData);

            long endTime = System.currentTimeMillis();
            System.out.println("Training completed in " + (endTime - startTime) / 1000.0 + " seconds");

        } catch (Exception e) {
            System.err.println("Error training Bagging: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Evaluate using 10-fold cross-validation
     */
    @Override
    public Evaluation evaluateModel() {
        Evaluation eval = null;
        try {
            if (trainData == null) {
                System.err.println("No training data loaded!");
                return null;
            }

            System.out.println("\n=== Evaluating Bagging Ensemble (10-Fold CV) ===");

            eval = new Evaluation(trainData);
            long startTime = System.currentTimeMillis();

            eval.crossValidateModel(baggingClassifier, trainData, 10, new Random(1));

            long endTime = System.currentTimeMillis();
            double evalTime = (endTime - startTime) / 1000.0;

            // Print results
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

            System.out.println("\n=== Performance Metrics ===");
            System.out.println("Accuracy:  " + String.format("%.2f%%", eval.pctCorrect()));
            System.out.println("Precision: " + String.format("%.4f", eval.weightedPrecision()));
            System.out.println("Recall:    " + String.format("%.4f", eval.weightedRecall()));
            System.out.println("F-Measure: " + String.format("%.4f", eval.weightedFMeasure()));
            System.out.println("Kappa:     " + String.format("%.4f", eval.kappa()));
            System.out.println("ROC Area:  " + String.format("%.4f", eval.weightedAreaUnderROC()));
            System.out.println("Evaluation time: " + evalTime + " seconds");

        } catch (Exception e) {
            System.err.println("Error evaluating Bagging: " + e.getMessage());
            e.printStackTrace();
        }
        return eval;
    }

    /**
     * Save model
     */
    @Override
    public void saveModel(String outputPath) {
        try {
            File file = new File(outputPath);
            file.getParentFile().mkdirs();

            SerializationHelper.write(outputPath, baggingClassifier);
            System.out.println("\nBagging model saved to: " + outputPath);

        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Load saved model
     */
    @Override
    public void loadModel(String modelPath) {
        try {
            baggingClassifier = (Bagging) SerializationHelper.read(modelPath);
            System.out.println("Bagging model loaded from: " + modelPath);

        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Getters
    public Classifier getEnsembleClassifier() {
        return baggingClassifier;
    }

    @Override
    public Instances getTrainData() {
        return trainData;
    }
}