package org.classpj.Models;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.File;
import java.util.Random;

/**
 * Step 3: Improved Classification
 * Extends WekaClassifier with ensemble methods to improve prediction accuracy.
 * This class implements advanced classification techniques:
 * 1. Bagging - Bootstrap Aggregating
 * 2. AdaBoost - Adaptive Boosting
 * 3. Voting - Majority Vote Ensemble
 * 4. Stacking - Meta-Learning Ensemble
 */
public class ImprovedClassifiers extends WekaClassifier {

    private Classifier ensembleClassifier;
    private String ensembleType;
    private Instances trainData;

    public ImprovedClassifiers(String ensembleType, String baseClassifierType) {
        super(baseClassifierType); // Initialize parent WekaClassifier
        this.ensembleType = ensembleType.toUpperCase();
        initializeEnsemble(baseClassifierType);
    }

    /**
     * Initialize the ensemble classifier based on type
     */
    private void initializeEnsemble(String baseClassifierType) {
        try {
            Classifier baseClassifier = createBaseClassifier(baseClassifierType);

            switch (ensembleType) {
                case "BAGGING":
                    Bagging bagging = new Bagging();
                    bagging.setClassifier(baseClassifier);
                    bagging.setNumIterations(10);
                    bagging.setBagSizePercent(100);
                    ensembleClassifier = bagging;
                    break;

                case "ADABOOST":
                    AdaBoostM1 adaBoost = new AdaBoostM1();
                    adaBoost.setClassifier(baseClassifier);
                    adaBoost.setNumIterations(10);
                    adaBoost.setUseResampling(true);
                    ensembleClassifier = adaBoost;
                    break;

                case "VOTING":
                    Vote vote = new Vote();
                    Classifier[] classifiers = new Classifier[4];
                    classifiers[0] = new J48();
                    classifiers[1] = new NaiveBayes();
                    classifiers[2] = new SMO();
                    classifiers[3] = new RandomForest();
                    vote.setClassifiers(classifiers);
                    vote.setCombinationRule(new weka.core.SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));
                    ensembleClassifier = vote;
                    break;

                case "STACKING":
                    Stacking stacking = new Stacking();
                    Classifier[] baseClassifiers = new Classifier[3];
                    baseClassifiers[0] = new J48();
                    baseClassifiers[1] = new NaiveBayes();
                    baseClassifiers[2] = new SMO();
                    stacking.setClassifiers(baseClassifiers);
                    stacking.setMetaClassifier(new RandomForest());
                    ensembleClassifier = stacking;
                    break;

                default:
                    System.out.println("Unknown ensemble type. Using Bagging as default.");
                    Bagging defaultBagging = new Bagging();
                    defaultBagging.setClassifier(baseClassifier);
                    ensembleClassifier = defaultBagging;
                    this.ensembleType = "BAGGING";
            }
        } catch (Exception e) {
            System.err.println("Error initializing ensemble: " + e.getMessage());
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
            case "SMO":
                return new SMO();
            case "RANDOMFOREST":
                RandomForest rf = new RandomForest();
                rf.setNumIterations(100);
                return rf;
            default:
                return new J48();
        }
    }

    /**
     * Load data - overrides parent
     */
    @Override
    public void loadData(String arffFilePath) {
        super.loadData(arffFilePath);
        this.trainData = super.getTrainData();
    }

    /**
     * Load Instances directly
     */
    public void loadData(Instances data) {
        this.trainData = new Instances(data);
        if (trainData.classIndex() == -1) {
            trainData.setClassIndex(trainData.numAttributes() - 1);
        }
        System.out.println("\n=== Data Loaded for " + ensembleType + " Ensemble ===");
        System.out.println("Number of instances: " + trainData.numInstances());
        System.out.println("Number of attributes: " + trainData.numAttributes());
        System.out.println("Class attribute: " + trainData.classAttribute().name());
    }

    /**
     * Train the ensemble classifier - overrides parent
     */
    @Override
    public void trainClassifier() {
        try {
            if (trainData == null) {
                System.err.println("No training data loaded!");
                return;
            }

            System.out.println("\n=== Training " + ensembleType + " Ensemble ===");
            long startTime = System.currentTimeMillis();

            ensembleClassifier.buildClassifier(trainData);

            long endTime = System.currentTimeMillis();
            System.out.println("Training completed in " + (endTime - startTime) / 1000.0 + " seconds");

        } catch (Exception e) {
            System.err.println("Error training ensemble: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Evaluate ensemble - overrides parent
     */
    @Override
    public Evaluation evaluateModel() {
        Evaluation eval = null;
        try {
            if (trainData == null) {
                System.err.println("No training data loaded!");
                return null;
            }

            System.out.println("\n=== Evaluating " + ensembleType + " Ensemble (10-Fold CV) ===");

            eval = new Evaluation(trainData);
            long startTime = System.currentTimeMillis();

            eval.crossValidateModel(ensembleClassifier, trainData, 10, new Random(1));

            long endTime = System.currentTimeMillis();
            double evalTime = (endTime - startTime) / 1000.0;

            // Print results
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

            System.out.println("\n=== Performance Metrics ===");
            System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
            System.out.println("Precision: " + String.format("%.4f", eval.weightedPrecision()));
            System.out.println("Recall: " + String.format("%.4f", eval.weightedRecall()));
            System.out.println("F-Measure: " + String.format("%.4f", eval.weightedFMeasure()));
            System.out.println("Kappa: " + String.format("%.4f", eval.kappa()));
            System.out.println("ROC Area: " + String.format("%.4f", eval.weightedAreaUnderROC()));
            System.out.println("Evaluation time: " + evalTime + " seconds");

        } catch (Exception e) {
            System.err.println("Error evaluating ensemble: " + e.getMessage());
            e.printStackTrace();
        }
        return eval;
    }

    /**
     * Save ensemble model - overrides parent
     */
    @Override
    public void saveModel(String outputPath) {
        try {
            File file = new File(outputPath);
            file.getParentFile().mkdirs();

            SerializationHelper.write(outputPath, ensembleClassifier);
            System.out.println("\n" + ensembleType + " ensemble model saved to: " + outputPath);

        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Load ensemble model - overrides parent
     */
    @Override
    public void loadModel(String modelPath) {
        try {
            ensembleClassifier = (Classifier) SerializationHelper.read(modelPath);
            System.out.println(ensembleType + " ensemble model loaded from: " + modelPath);

        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Getters
    public Classifier getEnsembleClassifier() {
        return ensembleClassifier;
    }

    public String getEnsembleType() {
        return ensembleType;
    }

    @Override
    public Instances getTrainData() {
        return trainData;
    }
}
