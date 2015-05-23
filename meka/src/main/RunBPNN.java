package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import meka.classifiers.multilabel.BPNN;
import meka.classifiers.multilabel.Evaluation;
import meka.core.F;
import meka.core.Result;
import weka.core.Instances;

public class RunBPNN {
	
	private final int nTarget;
	private final Instances train;
	private final Instances test;

	public RunBPNN(Instances train, Instances test, int nTarget) {
		this.train = mulanToMeka(train, nTarget);
		this.test = mulanToMeka(test, nTarget);
		this.nTarget = nTarget;
	}
	
	public void run() throws Exception {
		BPNN classifier = new CostumBPNN(1);
		classifier.setDebug(true);	
		Result r = Evaluation.evaluateModel(classifier, train, test);
		System.out.println(Result.getResultAsString(r));
	}
	
	public static Instances mulanToMeka(Instances data, int nClass) {
		Instances newData = F.mulan2meka(data, nClass);
		data.setClassIndex(nClass);
		return newData;
	}

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		Instances train = new Instances(reader);
		reader.close();
		
		reader = new BufferedReader(new FileReader(args[1]));
		Instances test = new Instances(reader);
		reader.close();
		
		RunBPNN runBPNN = new RunBPNN(train, test, Integer.parseInt(args[2]));
		runBPNN.run();

	}

}
