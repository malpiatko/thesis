package main;

import java.util.Random;

import meka.classifiers.multilabel.BPNN;

public class CostumBPNN extends BPNN {

	public CostumBPNN(int seed) {
		super();
		this.r = new Random(seed);
	}

}
