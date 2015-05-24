package main;

import java.util.Random;

import main.BPNN;

public class CostumBPNN extends BPNN {

	public CostumBPNN(int seed) {
		super();
		this.r = new Random(seed);
	}

}
