package org.classpj.Processors;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.JFrame;
public class ChartCreation {

    //HAVE TO BE NUMERIC CHART, THIS IS IMPORTANT PLEASE DO NOT USE THIS FOR ANYTHING ELSE
    public static void CreateNumericHistChart(Instances data, int attrIndex){
        double[] values = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            values[i] = data.instance(i).value(attrIndex);
        }
        HistogramDataset dataset = new HistogramDataset();
        dataset.addSeries(data.attribute(attrIndex).name(), values, 15);

        JFreeChart chart = ChartFactory.createHistogram(
                String.format("Histogram of %s", data.attribute(attrIndex).name()),
                data.attribute(attrIndex).name(),
                "Frequency",
                dataset
        );
        JFrame frame = new JFrame("Histogram");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    public static void CreateNominalBarChart(Instances data, int attrIndex){
        int numVals = data.attribute(attrIndex).numValues();
        double[] counts = new double[numVals];
        for (int i = 0; i < data.numInstances(); i++) {
            int valIndex = (int) data.instance(i).value(attrIndex);
            counts[valIndex]++;
        }

        DefaultCategoryDataset Catdataset = new DefaultCategoryDataset();

        for (int i = 0; i < numVals; i++) {
            Catdataset.addValue(counts[i], "Frequency", data.attribute(attrIndex).value(i));
        }

        JFreeChart barChart = ChartFactory.createBarChart(
                "Frequency Chart",
                "Category",
                "Count",
                Catdataset
        );
        JFrame frame = new JFrame("Bar Chart");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(barChart));
        frame.pack();
        frame.setVisible(true);

    }


}
