package org.classpj;

import org.classpj.Processors.ChartCreation;
import org.classpj.Processors.DataProcessing;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");

        data = DataProcessing.BasicPreprocessing(data);
        data = DataProcessing.ResamplingData(data);
        System.out.println(data.numInstances());

        Instances[] TrainTestSets = new Instances[2];
        TrainTestSets = DataProcessing.TrainTestSplit(data);
        System.out.println(TrainTestSets[0].numInstances());
//        ChartCreation.CreateNominalBarChart(data, 1);
//        System.out.println(data);
//        DataProcessing.ChiSquaredAttrSelector(data);
    }
}