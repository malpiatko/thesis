package likeability;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class TestFilter extends SimpleBatchFilter {
	
	private Attribute attributeA;
	private Attribute attributeB;

	TestFilter(Attribute a, Attribute b) {
		this.attributeA = a;
		this.attributeB = b;
	}
	
	@Override
	protected Instances process(Instances inst) throws Exception {
		RemoveWithValues partA = new RemoveWithValues();
		partA.setAttributeIndex(Integer.toString(attributeA.index()));
		partA.setInvertSelection(true);
		partA.setNominalIndices("0");
		Instances male = Filter.useFilter(inst, partA);
		return inst;
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
