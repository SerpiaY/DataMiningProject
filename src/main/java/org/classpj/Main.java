package org.classpj;

import org.classpj.Models.ImprovedClassifiers;
import org.classpj.Models.WekaClassifier;
import org.classpj.Processors.DataProcessing;
import org.classpj.Processors.ModelEvaluators;
import org.classpj.Processors.ModelEvaluators.ModelResult;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        // STEP 1: Load and preprocess data
        System.out.println("=== STEP 1: DATA PREPROCESSING ===");
        Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
        data.setClass(data.attribute("Heart Disease Status"));
        data = DataProcessing.BasicPreprocessing(data);
        data = DataProcessing.OversamplingData(data);

        // STEP 2: Standard Classifiers

        String filePath = "datasets/heart_disease.arff";

        System.out.println("\n=== STEP 2: STANDARD CLASSIFIERS ===");

        String[] classifierTypes = { "J48", "NaiveBayes" };

        // Store results for Step 4 comparison
        List<ModelResult> allResults = new ArrayList<>();

        for (String classifierType : classifierTypes) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Training and Evaluating: " + classifierType);
            System.out.println("=".repeat(80));

            // Create classifier
            WekaClassifier classifier = new WekaClassifier(classifierType);

            // Load preprocessed data
            classifier.loadData(filePath);

            // Train the classifier and measure time
            long trainStart = System.nanoTime();
            classifier.trainClassifier();
            long trainEnd = System.nanoTime();
            double trainTime = (trainEnd - trainStart) / 1_000_000_000.0;

            // Evaluate with 10-fold cross-validation and measure time
            long evalStart = System.nanoTime();
            Evaluation eval = classifier.evaluateModel();
            long evalEnd = System.nanoTime();
            double evalTime = (evalEnd - evalStart) / 1_000_000_000.0;

            // Store result for Step 4
            allResults.add(new ModelResult(classifierType, eval, trainTime, evalTime));

            // Save the trained model
            classifier.saveModel("models/" + classifierType + "_model.model");

            System.out.println("=".repeat(80));
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("STEP 2 COMPLETED - All base classifiers trained and evaluated!");
        System.out.println("=".repeat(80));

        // STEP 3: Improved Classification - Bagging Ensemble
        System.out.println("\n=== STEP 3: IMPROVED CLASSIFICATION - BAGGING ENSEMBLE ===");

        String[] baggingBaseClassifiers = { "J48", "NaiveBayes" };

        for (String baseClassifierType : baggingBaseClassifiers) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Bagging with Base Classifier: " + baseClassifierType);
            System.out.println("=".repeat(80));

            // Create Improved Classifier with Bagging
            ImprovedClassifiers improvedClassifier = new ImprovedClassifiers(baseClassifierType);

            // Load preprocessed data
            improvedClassifier.loadData(filePath);

            // Train the Bagging classifier and measure time
            long trainStart = System.nanoTime();
            improvedClassifier.trainClassifier();
            long trainEnd = System.nanoTime();
            double trainTime = (trainEnd - trainStart) / 1_000_000_000.0;

            // Evaluate with 10-fold cross-validation and measure time
            long evalStart = System.nanoTime();
            Evaluation eval = improvedClassifier.evaluateModel();
            long evalEnd = System.nanoTime();
            double evalTime = (evalEnd - evalStart) / 1_000_000_000.0;

            // Store result for Step 4
            String modelName = "BAGGING (" + baseClassifierType + ")";
            allResults.add(new ModelResult(modelName, eval, trainTime, evalTime));

            // Save the trained Bagging model
            improvedClassifier.saveModel("models/Bagging_" + baseClassifierType + "_model.model");

            System.out.println("=".repeat(80));
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("STEP 3 COMPLETED - Bagging ensemble improves upon base classifiers!");
        System.out.println("=".repeat(80));

        // STEP 4: Model Evaluation and Comparison Report
        // Generate comprehensive comparison using results from Steps 2 and 3
        ModelEvaluators.compareModels(allResults);

        // Generate detailed analysis and remarks
        ModelEvaluators.generateAnalysisReport(allResults);

        // Save results to CSV
        ModelEvaluators.saveResultsToCSV(allResults, "evaluation_results.csv");

        System.out.println("\n" + "=".repeat(100));
        System.out.println("STEP 4 COMPLETED - Comprehensive evaluation report generated!");
        System.out.println("=".repeat(100));
    }
}