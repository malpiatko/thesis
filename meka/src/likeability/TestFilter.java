package likeability;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class TestFilter extends SimpleBatchFilter {
	
	private Attribute a;
	private Attribute b;
	
	@Override
	protected Instances process(Instances inst) throws Exception {
		ArrayList<Instances> partitionsA = partition(inst, a);
		ArrayList<Instances> partitions = new ArrayList<Instances>();
		for(Instances data: partitionsA) {
			partitions.addAll(partition(data, b));
		}
		return inst;
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

	@Override
	protected Instances determineOutputFormat(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public static void main(String[] args) {
	     runFilter(new RemoveWithValues(), args);
	   }
}
