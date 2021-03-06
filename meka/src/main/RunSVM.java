package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import likeability.TestFilter;
import meka.core.A;
import meka.core.F;
import meka.core.LabelSet;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instances;

public class RunSVM {
	
	private final Instances train;
	private final Instances test;
	private final int nTarget;
	private final int compPrec = 4;
	private Map<Attribute, Double> cValues;
	
	private int seed = 1;
	private int folds = 10;
	
	RunSVM(Instances train, Instances test, String attr) {
		this.train = train;
		this.test = test;
		this.nTarget = Integer.parseInt(attr);
		
		this.train.setClassIndex(nTarget);
		this.test.setClassIndex(nTarget);
		cValues = new LinkedHashMap<Attribute, Double>();
	}
	
	RunSVM(Instances train, Instances test, String attr, boolean mulan) {
		this(train, test, attr);
		if(mulan){
			F.mulan2meka(train, nTarget);
			F.mulan2meka(test, nTarget);
			train.setClassIndex(nTarget);
			test.setClassIndex(nTarget);
		}
	}
	
	public void CV() throws Exception {
		for(int idx = 0; idx < nTarget; idx++) {
			Instances data = removeLabels(train, idx);
			CVSingle(data);
		}
	}
	
	private void CVSingle(Instances data) throws Exception {
		Attribute a = data.classAttribute();
		System.out.print("Calculating best c for " + a.name() + " ");
		double maxRecall = 0;
		double c = 0;
		for(int i = compPrec; i >= 0; i--){
			Evaluation eval = new Evaluation(data);
			SMO classifier = new SMO();
			double currC = Math.pow(10, -i);
			classifier.setC(currC);
			eval.crossValidateModel(classifier, data, folds, new Random(seed));
			double uar = getUAR(eval, a.numValues());
			if (uar > maxRecall){
				maxRecall = uar;
				c = currC;
			}
		}
		System.out.println(c+ "max recall " + maxRecall);
		cValues.put(a, c);		
	}
	
	private double getUAR(Evaluation eval, int nClass){
		double uar = 0;
		for(int i = 0; i < nClass; i++) {
			uar += eval.recall(i);
		}
		return uar/nClass;
	}
	
	private void getBaseline() throws Exception {
		for(Attribute a: cValues.keySet()){
			//Prepare train and test set
			int idx = train.attribute(a.name()).index();
			Instances newTrain = removeLabels(train, idx);
			Instances newTest = removeLabels(test, idx);
			
			//Prepare classifier
			SMO classifier = new SMO();
			classifier.setC(cValues.get(a));
			classifier.buildClassifier(newTrain);
			
			//Evaluate performance
			Evaluation eval = new Evaluation(newTrain);
			eval.evaluateModel(classifier, newTest);
			double uar = getUAR(eval, a.numValues());
			System.out.println("Target " + a.name() + " UAR=" + uar);
		}		
	}
	
	/*
	 * Removes all class labels except the one given at idx
	 */
	private Instances removeLabels(Instances data, int idx) throws Exception {
		Instances newData = F.keepLabels(data, nTarget, new int[]{idx});
		newData.setClassIndex(0);
		newData.deleteAttributeType(Attribute.STRING);
		return newData;
	}

	/**
	 * @param args train_file, test_file, nAttr
	 * Runs the SVM classifier on each class, first choosing
	 * the best C
	 * Assumes the arff file is in MULAN format, ie. the classes are last
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		//Load train set
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		Instances train = new Instances(reader);
		reader.close();
		//Load test set
		reader = new BufferedReader(new FileReader(args[1]));
		Instances test = new Instances(reader);
		reader.close();
		RunSVM runSVM = new RunSVM(train, test, args[2], true);
		runSVM.CV();
		runSVM.getBaseline();
	}

}
