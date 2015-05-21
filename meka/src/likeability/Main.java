package likeability;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.core.Attribute;
import weka.core.Instances;

public class Main {
	
	Instances train;
	Instances test;

	public static void main(String[] args) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(args[1]));
		Instances data = new Instances(reader);
		reader.close();
		
		
	}
	
	public static void getTest(Instances data) throws Exception {
	}

}
