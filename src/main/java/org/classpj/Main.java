package org.classpj;

import org.classpj.Models.ImprovedClassifiers;
import org.classpj.Models.WekaClassifier;
import org.classpj.Processors.ChartCreation;
import org.classpj.Processors.DataProcessing;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
         // DATA
         Instances data =
         DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
         data.setClass(data.attribute("Heart Disease Status"));
//
//         // PROCESS DATA
//         data = DataProcessing.BasicPreprocessing(data);
//         // SMOTE OVERSAMPLING
//         data = DataProcessing.OversamplingData(data);
//
//         // CREATE TRAIN AND TEST SETS
//         Instances[] TrainTestSets = new Instances[2];
//         TrainTestSets = DataProcessing.TrainTestSplit(data);
//         System.out.println(TrainTestSets[0].numInstances());
////         ChartCreation.CreateNominalBarChart(data, 1);
//          DataProcessing.ChiSquaredAttrSelector(data);

         //run J48 classifier
//         System.out.println("\n=== STEP 2: NB CLASSIFICATION ===\n");
//         WekaClassifier j48Classifier = new WekaClassifier("NAIVEBAYES");
//         j48Classifier.loadData("datasets/heart_disease.arff");
//         j48Classifier.trainClassifier();
//         j48Classifier.evaluateModel();
//         j48Classifier.saveModel("models/NB_model.model");

        // Test ImprovedClassifiers with different ensemble methods (Step 3)
//        System.out.println("\n" + "=".repeat(80));
//        System.out.println("TESTING IMPROVED CLASSIFIERS - ENSEMBLE METHODS");
//        System.out.println("=".repeat(80));
//
//        String[] ensembleMethods = { "BAGGING" };
//
//        for (String ensembleType : ensembleMethods) {
//            System.out.println("\n" + "=".repeat(80));
//            ImprovedClassifiers improved = new ImprovedClassifiers(ensembleType, "NAIVEBAYES");
//            improved.loadData("datasets/heart_disease.arff");
//            improved.trainClassifier();
//            improved.evaluateModel();
//            improved.saveModel("models/" + ensembleType + "_model.model");
//            System.out.println("=".repeat(80));
//        }
//
//        System.out.println("\n" + "=".repeat(80));
//        System.out.println("ALL ENSEMBLE METHODS TESTED SUCCESSFULLY!");
//        System.out.println("=".repeat(80));
    }
}