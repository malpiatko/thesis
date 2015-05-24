package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MULAN;
import meka.core.F;
import meka.core.Result;
import mulan.classifier.neural.model.ActivationLinear;
import mulan.classifier.neural.model.BasicNeuralNet;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

public class RunBPNN {
	
	private final int seed = 1;
	private final int nTarget;
	private final Instances train;
	private final Instances test;

	public RunBPNN(Instances train, Instances test, int nTarget, boolean toPrep) throws Exception {
		if(toPrep) {
			this.train = mulanToMeka(train, nTarget);
			this.test = mulanToMeka(test, nTarget);
		} else {
			this.train = train;
			this.test = test;
		}
		train.setClassIndex(nTarget);
		test.setClassIndex(nTarget);
		this.train.randomize(new Random(seed));
		this.nTarget = nTarget;
	}
	
	public void run(int e) throws Exception {
		BPNN classifier = new CostumBPNN(1);
		//classifier.setDebug(true);
		classifier.setE(e);
		Result r = Evaluation.evaluateModel(classifier, train, test);
		System.out.println(Result.getResultAsString(r));
	}
	
	public BPNN buildClassifier() throws Exception {
		BPNN classifier = new CostumBPNN(1);
		//classifier.setDebug(true);
		classifier.buildClassifier(train);
		return classifier;
	}
	
	public static Instances mulanToMeka(Instances data, int nClass) {
		Instances newData = F.mulan2meka(data, nClass);
		return newData;
	}

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		Instances train = new Instances(reader);
		reader.close();
		
		reader = new BufferedReader(new FileReader(args[1]));
		Instances test = new Instances(reader);
		reader.close();
		
		RunBPNN runBPNN = new RunBPNN(train, test, Integer.parseInt(args[2]), false);
		/*for(int i = 1; i < 41; i++) {
			runBPNN.run(i);
			System.out.println(i);
			System.in.read();
		}*/
		//runBPNN.run(1000);
		//runBPNN.buildClassifier();
		MULAN mulan = new MULAN();
		mulan.setMethod("BPMLL");
		mulan.setDebug(true);
		System.out.println(mulan.listOptions());
		Result result = Evaluation.evaluateModel(mulan, runBPNN.train, runBPNN.test);
		System.out.println(Result.getResultAsString(result));
		//BasicNeuralNet mulanNN = new BasicNeuralNet(new int[]{102,10,2}, 1, ActivationLinear.class, new Random(1));
		

	}

}
