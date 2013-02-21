package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;

import java.util.Collection;

public class SentSumSimilarityMeasure
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;

	public SentSumSimilarityMeasure(int n)
	{
		// The configuration parameter is not used right now and intended for illustration purposes only.
		this.n = n;
	}
	
	@Override
	public double getSimilarity(Collection<String> strings,
			Collection<String> strings2)
		throws SimilarityException
	{
        int l1 = strings.iterator().next().split(" ").length;
        int l2 = strings2.iterator().next().split(" ").length;

        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();

        int sum1 = 0;
        int sum2 = 0;


        for (char c: first_string.toCharArray())
        {
            int i = c;
            sum1 += i;

        }

        for (char c: second_string.toCharArray())
        {
            int i = c;
            sum2 += i;

        }

        int diff = Math.abs(sum2 - sum1);


        return diff;  //To change body of implemented methods use File | Settings | File Templates.
		// Your similarity computation goes here.

	}

}