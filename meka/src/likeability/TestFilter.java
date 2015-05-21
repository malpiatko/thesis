package likeability;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.Resample;

@SuppressWarnings("serial")
public class TestFilter extends SimpleBatchFilter {
	
	private Attribute a;
	private Attribute b;
	private int sampleSizePercent = 15;
	private boolean invert = false;
	private int seed = 1;
	
	@Override
	protected Instances process(Instances inst) throws Exception {
		ArrayList<Instances> partitionsA = partition(inst, a);
		ArrayList<Instances> partitions = new ArrayList<Instances>();
		for(Instances data: partitionsA) {
			partitions.addAll(partition(data, b));
		}
		
		return getTestSet(partitions);
	}
	
	/*
	 * Partitions the data so that there's only one nominal value of the
	 * attribute a in one partition.
	 */
	private ArrayList<Instances> partition(Instances data, Attribute a) throws Exception {
		ArrayList<Instances> instances = new ArrayList<Instances>();
		for (int i = 0; i < a.numValues(); i++){
			RemoveWithValues rm = new RemoveWithValues();
			rm.setAttributeIndex(Integer.toString(a.index()));
			rm.setInvertSelection(true);
			rm.setNominalIndices(Integer.toString(i));
			instances.add(Filter.useFilter(data, rm));
		}
		return instances;
	}
	
	private Instances getTestSet(List<Instances> insts) throws Exception {
		Instances output = new Instances(insts.get(0), 0);
		Resample filter = new Resample();
		filter.setRandomSeed(seed);
		filter.setNoReplacement(true);
		filter.setInvertSelection(invert);
		filter.setSampleSizePercent(sampleSizePercent);
		for(Instances inst: insts) {
			output.addAll(Filter.useFilter(inst, filter));
		}
		return output;
		
	}

	@Override
	protected Instances determineOutputFormat(Instances arg) throws Exception {
		return new Instances(arg, 0);
	}

	@Override
	public String globalInfo() {
		return "A filter which partitions the data so that each partition contains"
				+ " only instances with one value of attribute a and b, then takes "
				+ "a random subset of values from each partition and merges them to"
				+ " produce the final set.";
	}
	
	public static void main(String[] args) {
	     runFilter(new RemoveWithValues(), args);
	}
	
	public Attribute getA() {
		return a;
	}

	public void setA(Attribute a) {
		this.a = a;
	}

	public Attribute getB() {
		return b;
	}

	public void setB(Attribute b) {
		this.b = b;
	}

	public int getSampleSizePercent() {
		return sampleSizePercent;
	}

	public void setSampleSizePercent(int sampleSizePercent) {
		this.sampleSizePercent = sampleSizePercent;
	}

	public boolean isInvert() {
		return invert;
	}

	public void setInvert(boolean invert) {
		this.invert = invert;
	}

	public int getSeed() {
		return seed;
	}

	public void setSeed(int seed) {
		this.seed = seed;
	}
}
