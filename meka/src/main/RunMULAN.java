package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import meka.core.F;
import mulan.classifier.neural.BPMLL;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.Measure;
import weka.core.Instances;

public class RunMULAN {
	
	private static long seed = 1;
	
	private double defaultRate = 0.01;
	private int defaultNodes = 62;
	private int defaultEpochs = 200;

	RunMULAN(double defaultRate, int defaultNodes, int defaultEpochs) {
		this.defaultRate = defaultRate;
		this.defaultNodes = defaultNodes;
		this.defaultEpochs = defaultEpochs;
	}
	
	public RunMULAN() {
	}

	public BPMLL getDefaultClassifier() {
		BPMLL classifier = new BPMLL(seed);
		classifier.setLearningRate(defaultRate);
		classifier.setHiddenLayers(new int[]{defaultNodes});
		classifier.setTrainingEpochs(defaultEpochs);
		return classifier;
	}
	
	public Evaluation runEvaluation(BPMLL classifier, MultiLabelInstances test) throws IllegalArgumentException, Exception {
		Evaluator eval = new Evaluator();
		ArrayList<Measure> measures = new ArrayList<Measure>();
		measures.add(new MacroRecall(test.getNumLabels()));
		return eval.evaluate(classifier, test, measures);
	}
	
	public void testLearningRate(MultiLabelInstances train, MultiLabelInstances test) throws IllegalArgumentException, Exception {
		for(int i = 0; i < 5; i++) {
			double pow = Math.pow(10,-i);
			BPMLL classifier = getDefaultClassifier();
			classifier.setLearningRate(pow);
			classifier.build(train);
			Evaluation eval =runEvaluation(classifier, test);
			System.out.println("Rate=" + pow + "\n" + eval.toString());
		}
	}
	
	public void testNodes(MultiLabelInstances train, MultiLabelInstances test) throws IllegalArgumentException, Exception {
		int nodes = 1;
		for(int i = 1; i <= 6; i++) {
			nodes *= 2;
			BPMLL classifier = getDefaultClassifier();
			classifier.setHiddenLayers(new int[]{nodes});
			classifier.build(train);
			Evaluation eval =runEvaluation(classifier, test);
			System.out.println("Nodes=" + nodes + "\n" + eval.toString());
		}
	}
	
	public void testEpochs(MultiLabelInstances train, MultiLabelInstances test) throws IllegalArgumentException, Exception {
		int epochs = 1;
		for(int i = 1; i <= 10; i++) {
			epochs = i*100;
			BPMLL classifier = getDefaultClassifier();
			classifier.setTrainingEpochs(epochs);
			classifier.build(train);
			Evaluation eval =runEvaluation(classifier, test);
			System.out.println("Epochs=" + epochs + "\n" + eval.toString());
		}
	}
	
	public void crossV(MultiLabelInstances train, int folds) {
		System.out.println("Running cross-validation test");
		BPMLL classifier = getDefaultClassifier();
		Evaluator eval = new Evaluator();
		ArrayList<Measure> measures = new ArrayList<Measure>();
		measures.add(new MacroRecall(train.getNumLabels()));
		MultipleEvaluation evaluation = eval.crossValidate(classifier, train, measures, folds);
		System.out.println(evaluation.toString());
	}
	
	public void runStandard(MultiLabelInstances train, MultiLabelInstances test) throws Exception{
		BPMLL classifier = getDefaultClassifier();
		classifier.build(train);
		Evaluation eval =runEvaluation(classifier, test);
		System.out.println(eval.toString());
	}

	public static void main(String[] args) throws Exception {
		
		int nTarget = Integer.parseInt(args[2]);
		MultiLabelInstances train = new MultiLabelInstances(args[0], nTarget);
		MultiLabelInstances test = new MultiLabelInstances(args[1], nTarget);
		
		RunMULAN experiment = new RunMULAN();
		experiment.crossV(train, 10);
		experiment.runStandard(train, test);
	}

}
