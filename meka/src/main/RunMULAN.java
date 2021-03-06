package main;

import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;

import mulan.classifier.neural.BPMLL;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;

public class RunMULAN {
	
	private static long seed = 1;
	
	private double defaultRate = 0.05;
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
		eval.setSeed((int) seed);
		ArrayList<Measure> measures = new ArrayList<Measure>();
		measures.add(new UARMulan(test.getNumLabels()));
		return eval.evaluate(classifier, test, measures);
	}
	
	public void testLearningRate(MultiLabelInstances train) throws IllegalArgumentException, Exception {
		for(int i = 0; i < 5; i++) {
			defaultRate = Math.pow(10,-i);
			crossV(train, 5);
		}
	}
	
	public void testNodes(MultiLabelInstances train) throws IllegalArgumentException, Exception {
		defaultNodes = 1;
		for(int i = 1; i <= 6; i++) {
			defaultNodes *= 2;
			crossV(train, 5);
		}
	}
	
	public void testEpochs(MultiLabelInstances train) throws IllegalArgumentException, Exception {
		for(int i = 1; i <= 6; i++) {
			defaultEpochs = i*100;
			crossV(train, 5);
		}
	}
	
	public void crossV(MultiLabelInstances train, int folds) {
		System.out.println("Running cross-validation test");
		BPMLL classifier = getDefaultClassifier();
		Evaluator eval = new Evaluator();
		eval.setSeed((int) seed);
		ArrayList<Measure> measures = new ArrayList<Measure>();
		measures.add(new UARMulan(train.getNumLabels()));
		MultipleEvaluation evaluation = eval.crossValidate(classifier, train, measures, folds);
		System.out.println(toString());
		System.out.println(evaluation.toString());
	}
	
	public void runStandard(MultiLabelInstances train, MultiLabelInstances test) throws Exception{
		BPMLL classifier = getDefaultClassifier();
		classifier.build(train);
		Evaluation eval =runEvaluation(classifier, test);
		System.out.println(toString());
		System.out.println(eval.toString());
	}
	
	@Override
	public String toString() {
		return "Rate: " + defaultRate + ",nodes: " + defaultNodes + ", epochs: " + defaultEpochs;
		
	}
	
	public static void runFullExperiment(MultiLabelInstances train, double rate) throws IllegalArgumentException, Exception {
		System.setOut(new PrintStream(new File("output.txt")));
		for(int nodes = 2; nodes <= 64; nodes*=2) {
			RunMULAN experiment = new RunMULAN(rate, nodes, 200);	
			experiment.testEpochs(train);
		}
	}

	public static void main(String[] args) throws Exception {
		
		int nTarget = Integer.parseInt(args[2]);
		MultiLabelInstances train = new MultiLabelInstances(args[0], nTarget);
		MultiLabelInstances test = new MultiLabelInstances(args[1], nTarget);
		
		RunMULAN experiment = new RunMULAN(0.01, 64, 200);
		//experiment.crossV(train, 5);
		//experiment.testEpochs(train);
		//experiment.testLearningRate(train);
		experiment.runStandard(train, test);
		//runFullExperiment(train, 0.01);
	}

}
