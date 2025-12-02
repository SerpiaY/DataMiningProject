package org.classpj.Processors;

import weka.classifiers.Evaluation;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Step 4: Model Evaluation and Report (20 pts)
 * - Compares performance of classification models
 * - Provides remarks on experimental results
 * - Considers run-time of building models and making predictions
 */
public class ModelEvaluators {

    private static final DecimalFormat df = new DecimalFormat("#.####");
    private static final DecimalFormat df2 = new DecimalFormat("#.##");

    /**
     * Store evaluation results for a single model
     */
    public static class ModelResult {
        public String modelName;
        public double accuracy;
        public double precision;
        public double recall;
        public double fMeasure;
        public double kappa;
        public double rocArea;
        public double trainTime;
        public double evalTime;
        public Evaluation evaluation;

        public ModelResult(String name, Evaluation eval, double trainTime, double evalTime) {
            this.modelName = name;
            this.evaluation = eval;
            this.accuracy = eval.pctCorrect();
            this.precision = eval.weightedPrecision();
            this.recall = eval.weightedRecall();
            this.fMeasure = eval.weightedFMeasure();
            this.kappa = eval.kappa();
            this.rocArea = eval.weightedAreaUnderROC();
            this.trainTime = trainTime;
            this.evalTime = evalTime;
        }
    }

    /**
     * Generate comprehensive comparison report for all models
     */
    public static void compareModels(List<ModelResult> results) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("MODEL COMPARISON REPORT");
        System.out.println("=".repeat(100));
        System.out.println();

        // Print header
        System.out.printf("%-20s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s%n",
                "Model", "Accuracy", "Precision", "Recall", "F-Measure", "Kappa", "ROC Area", "Train(s)", "Eval(s)");
        System.out.println("-".repeat(150));

        // Print each model's results
        for (ModelResult result : results) {
            System.out.printf("%-20s | %9.2f%% | %10s | %10s | %10s | %10s | %10s | %10s | %10s%n",
                    result.modelName,
                    result.accuracy,
                    df.format(result.precision),
                    df.format(result.recall),
                    df.format(result.fMeasure),
                    df.format(result.kappa),
                    df.format(result.rocArea),
                    df2.format(result.trainTime),
                    df2.format(result.evalTime));
        }
        System.out.println("-".repeat(150));

        // Find best performers
        ModelResult bestAccuracy = results.stream().max(Comparator.comparingDouble(r -> r.accuracy)).orElse(null);
        ModelResult bestFMeasure = results.stream().max(Comparator.comparingDouble(r -> r.fMeasure)).orElse(null);
        ModelResult bestROC = results.stream().max(Comparator.comparingDouble(r -> r.rocArea)).orElse(null);
        ModelResult fastestTrain = results.stream().min(Comparator.comparingDouble(r -> r.trainTime)).orElse(null);
        ModelResult fastestEval = results.stream().min(Comparator.comparingDouble(r -> r.evalTime)).orElse(null);

        System.out.println("\n=== BEST PERFORMERS ===");
        if (bestAccuracy != null) {
            System.out.printf("Highest Accuracy:     %s (%.2f%%)%n", bestAccuracy.modelName, bestAccuracy.accuracy);
        }
        if (bestFMeasure != null) {
            System.out.printf("Highest F-Measure:    %s (%.4f)%n", bestFMeasure.modelName, bestFMeasure.fMeasure);
        }
        if (bestROC != null) {
            System.out.printf("Highest ROC Area:     %s (%.4f)%n", bestROC.modelName, bestROC.rocArea);
        }
        if (fastestTrain != null) {
            System.out.printf("Fastest Training:     %s (%.2fs)%n", fastestTrain.modelName, fastestTrain.trainTime);
        }
        if (fastestEval != null) {
            System.out.printf("Fastest Evaluation:   %s (%.2fs)%n", fastestEval.modelName, fastestEval.evalTime);
        }
    }

    /**
     * Generate detailed analysis and remarks on experimental results
     */
    public static void generateAnalysisReport(List<ModelResult> results) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("EXPERIMENTAL RESULTS ANALYSIS & REMARKS");
        System.out.println("=".repeat(100));

        // Calculate averages
        double avgAccuracy = results.stream().mapToDouble(r -> r.accuracy).average().orElse(0);
        double avgTrainTime = results.stream().mapToDouble(r -> r.trainTime).average().orElse(0);
        double avgEvalTime = results.stream().mapToDouble(r -> r.evalTime).average().orElse(0);

        // Find best overall model
        ModelResult bestOverall = results.stream()
                .max(Comparator.comparingDouble(r -> r.accuracy + r.fMeasure + r.rocArea))
                .orElse(null);

        if (bestOverall != null) {
            System.out.println("\n=== OVERALL BEST MODEL ===");
            System.out.println("Model: " + bestOverall.modelName);
            System.out.printf(
                    "Reasoning: Balanced performance across accuracy (%.2f%%), F-Measure (%.4f), and ROC Area (%.4f)%n",
                    bestOverall.accuracy, bestOverall.fMeasure, bestOverall.rocArea);
        }

        System.out.println("\n=== KEY OBSERVATIONS ===");
        System.out.println();
        System.out.printf("1. Accuracy Analysis:%n");
        System.out.printf("   - Average Accuracy: %.4f%%%n", avgAccuracy);
        for (ModelResult result : results) {
            if (result.accuracy > avgAccuracy) {
                System.out.printf("   - %s performs above average (+%.4f%%)%n",
                        result.modelName, result.accuracy - avgAccuracy);
            }
        }

        System.out.println();
        System.out.printf("2. Runtime Performance:%n");
        System.out.printf("   - Average Training Time: %.2fs%n", avgTrainTime);
        System.out.printf("   - Average Evaluation Time: %.2fs%n", avgEvalTime);
        for (ModelResult result : results) {
            if (result.trainTime < avgTrainTime) {
                System.out.printf("   - %s trains faster than average (%.2fs)%n",
                        result.modelName, result.trainTime);
            }
        }

        // Compare base classifiers vs ensemble methods
        System.out.println();
        System.out.println("3. Ensemble Methods vs Base Classifiers:");
        List<ModelResult> baseModels = new ArrayList<>();
        List<ModelResult> ensembleModels = new ArrayList<>();

        for (ModelResult result : results) {
            if (result.modelName.contains("BAGGING")) {
                ensembleModels.add(result);
            } else {
                baseModels.add(result);
            }
        }

        if (!ensembleModels.isEmpty() && !baseModels.isEmpty()) {
            double avgBaseAccuracy = baseModels.stream().mapToDouble(r -> r.accuracy).average().orElse(0);
            double avgEnsembleAccuracy = ensembleModels.stream().mapToDouble(r -> r.accuracy).average().orElse(0);
            double improvement = avgEnsembleAccuracy - avgBaseAccuracy;

            System.out.printf("   - Base Classifiers Average Accuracy: %.2f%%%n", avgBaseAccuracy);
            System.out.printf("   - Ensemble Methods Average Accuracy: %.2f%%%n", avgEnsembleAccuracy);
            if (improvement > 0) {
                System.out.printf("   - Ensemble Improvement: +%.2f%%%n", improvement);
            }
        }

        System.out.println();
        System.out.println("=== RECOMMENDATIONS ===");

        ModelResult bestAccuracy = results.stream().max(Comparator.comparingDouble(r -> r.accuracy)).orElse(null);
        ModelResult fastestTotal = results.stream()
                .min(Comparator.comparingDouble(r -> r.trainTime + r.evalTime))
                .orElse(null);

        if (bestAccuracy != null) {
            System.out.println("1. For Production Use: " + bestAccuracy.modelName);
            System.out.println("   - Provides best balance of accuracy and performance");
        }

        if (fastestTotal != null) {
            System.out.println();
            System.out.println("2. For Real-Time Applications: " + fastestTotal.modelName);
            System.out.printf("   - Fastest training and prediction time%n");
            System.out.printf("   - Total time: %.2fs%n", fastestTotal.trainTime + fastestTotal.evalTime);
        }

        System.out.println();
        System.out.println("=".repeat(100));
    }

    /**
     * Save results to CSV file for further analysis
     */
    public static void saveResultsToCSV(List<ModelResult> results, String filename) {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(filename)) {
            // Write header
            writer.println("Model,Accuracy(%),Precision,Recall,F-Measure,Kappa,ROC_Area,Train_Time(s),Eval_Time(s)");

            // Write data
            for (ModelResult result : results) {
                writer.printf("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f%n",
                        result.modelName,
                        result.accuracy,
                        result.precision,
                        result.recall,
                        result.fMeasure,
                        result.kappa,
                        result.rocArea,
                        result.trainTime,
                        result.evalTime);
            }

            System.out.println("\nResults saved to: " + filename);
        } catch (Exception e) {
            System.err.println("Error saving results to CSV: " + e.getMessage());
        }
    }
}