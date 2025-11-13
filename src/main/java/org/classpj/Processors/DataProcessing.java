package org.classpj.Processors;

import weka.attributeSelection.*;
import weka.core.*;
import java.io.*;

import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.*;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.attributeSelection.ChiSquaredAttributeEval;

public class DataProcessing {

    public static Instances ReadAndReturnDataLocal(String datapath) throws IOException {
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
            return data;
        }
    }

    public static Instances BasicPreprocessing(Instances dataset) throws Exception {
        Filter filter = new ReplaceMissingValues();
        filter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, filter);
        return dataset;
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
}
