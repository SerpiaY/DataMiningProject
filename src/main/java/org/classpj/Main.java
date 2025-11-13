package org.classpj;

import org.classpj.Processors.DataProcessing;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        Instances data = DataProcessing.ReadAndReturnDataLocal("datasets/heart_disease.csv");
        data.setClass(data.attribute("Heart Disease Status"));
        data = DataProcessing.BasicPreprocessing(data);
//        System.out.println(data);
        DataProcessing.ChiSquaredAttrSelector(data);
    }
}