package org.classpj;

import org.classpj.Processors.ChartCreation;
import org.classpj.Processors.DataProcessing;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        //READ DATA
        Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
        //PROCESS DATA
        data = DataProcessing.BasicPreprocessing(data);
        //SMOTE OVERSAMPLING
        data = DataProcessing.OversamplingData(data);

        //CREATE TRAIN AND TEST SETS
        Instances[] TrainTestSets = new Instances[2];
        TrainTestSets = DataProcessing.TrainTestSplit(data);
        System.out.println(TrainTestSets[0].numInstances());
        ChartCreation.CreateNominalBarChart(data, 1);
//        DataProcessing.ChiSquaredAttrSelector(data);
    }
}