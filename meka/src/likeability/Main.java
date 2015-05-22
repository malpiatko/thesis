package likeability;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Main {
	
	Instances train;
	Instances test;

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		Instances data = new Instances(reader);
		reader.close();
		TestFilter filter = new TestFilter();
		filter.setA(data.attribute("gender"));
		filter.setB(data.attribute("age"));
		filter.setInputFormat(data);
		Instances test = Filter.useFilter(data, filter);
		filter.setInvert(true);
		filter.setInputFormat(data);
		Instances train = Filter.useFilter(data, filter);
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File(args[1]));
		saver.writeBatch();
		
		saver.setInstances(test);
		saver.setFile(new File(args[2]));
		saver.writeBatch();
		
	}

}
