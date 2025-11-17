package org.classpj.Processors;

import weka.attributeSelection.*;
import weka.core.*;
import java.io.*;
import java.util.Objects;

import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.*;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class DataProcessing {

    public static Instances ReadAndReturnDataLocal(String datapath) throws Exception {
        ClassLoader classLoader = DataProcessing.class.getClassLoader();
        InputStream inputStream = classLoader.getResourceAsStream(datapath);
        if (inputStream == null) {
            throw new IllegalArgumentException("file not found!: " + datapath);
        } else {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(datapath));
            Instances data = loader.getDataSet();
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File("datasets/heart_disease.arff"));
            saver.writeBatch();
            data.setClass(data.attribute("Heart Disease Status"));
            System.out.println("Data Loaded from path: "+ datapath);
            System.out.println("Number of instances: " + data.numInstances());
            System.out.println("Number of attributes: " + data.numAttributes());
            System.out.println("Class attribute: " + data.classAttribute().name());
            System.out.println("Summary " + data.toSummaryString());
            System.out.println("=================-----------------------================= ");
            for(int i = 0; i < data.numAttributes(); i++){
                if(data.attribute(i).isNumeric()){
                    System.out.println("Variance for "+ data.attribute(i).name() + ": " + data.variance(i));
                    System.out.println("Mean for "+ data.attribute(i).name() + ": " + data.meanOrMode(i));
                    System.out.println("=================-----------------------================= ");
                }
            }

            return data;
        }
    }


    public static Instances BasicPreprocessing(Instances data) throws Exception {
        int outlierCount = 0;
        int extremeCount = 0;
        System.out.println("Start Filtering");
        System.out.println("Total Attr before filtering: " +  data.numInstances());

        Filter filter = new ReplaceMissingValues();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        filter = new RemoveDuplicates();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        InterquartileRange ifilter = new InterquartileRange();
        ifilter.setInputFormat(data);
        ifilter.setAttributeIndices("first-last");
        data = Filter.useFilter(data, ifilter);
        filter = new NominalToBinary();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);


        System.out.println("Outliers & Extreme Values using IQR");
        for(int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).value(30) != 0) {
                outlierCount++;
            }
            if (data.instance(i).value(31) != 0) {
                extremeCount++;
            }
        }
        System.out.println("                Outliers Count: " + outlierCount);
        System.out.println("                Extremes Count: " + extremeCount);

        System.out.println("Total Attr after filtering: " +  data.numInstances());
        return data;
    }
    public static void PearsonCorrAttrSelector(Instances dataset) throws Exception {
        Filter filter = new NominalToBinary();
        filter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, filter);

        ASEvaluation evaluator = new CorrelationAttributeEval();
        ASSearch search = new Ranker();

        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(dataset);

        int[] selectedAttributes = selector.selectedAttributes();
        for (int idx : selectedAttributes) {
            System.out.println(idx + ": " + dataset.attribute(idx).name());
        }
    }

    public static void ChiSquaredAttrSelector(Instances dataset) throws Exception {
        Filter filter = new NumericToNominal();
        filter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, filter);
        ASEvaluation evaluator = new ChiSquaredAttributeEval();
        ASSearch search = new Ranker();

        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(dataset);

        int[] selectedAttributes = selector.selectedAttributes();
        for (int idx : selectedAttributes) {
            System.out.println(idx + ": " + dataset.attribute(idx).name());
        }
    }
    public static Instances OversamplingData(Instances data) throws Exception {
        SMOTE filter = new SMOTE();
        filter.setInputFormat(data);
        filter.setPercentage(200);
        filter.setNearestNeighbors(8);
        data = Filter.useFilter(data, filter);
        return data;
    }

    public static Instances ResamplingData(Instances data) throws Exception {
        Resample filter = new Resample();
        filter.setInputFormat(data);
        filter.setSeed(423);
        data = Filter.useFilter(data, filter);
        return data;
    }

    //0 IS FOR TRAIN SETS, 1 IS FOR TEST SETS
    public static Instances[] TrainTestSplit(Instances data) throws Exception {
        StratifiedRemoveFolds fold = new StratifiedRemoveFolds();
        fold.setNumFolds(10);
        fold.setSeed(4235456);
        fold.setInvertSelection(true);
        fold.setInputFormat(data);
        Instances train = Filter.useFilter(data, fold);

        fold.setInvertSelection(false);
        Instances test = Filter.useFilter(data, fold);
        Instances[] traintest = new Instances[2];
        traintest[0] = train;
        traintest[1] = test;
        return traintest;
    }

}
