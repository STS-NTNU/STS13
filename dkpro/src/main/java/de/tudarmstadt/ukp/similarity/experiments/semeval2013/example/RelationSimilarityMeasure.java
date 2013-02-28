package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;

import java.util.Collection;

import relex.RelationExtractor;
import relex.entity.EntityMaintainer;


public class RelationSimilarityMeasure
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;
    private RelationExtractor rel;

	public RelationSimilarityMeasure(int n, RelationExtractor rel)
	{                                                                          		// The configuration parameter is not used right now and intended for illustration purposes only.
		this.n = n;
        this.rel = rel;
	}

    public double getSimilarity(Collection<String> strings,
			Collection<String> strings2)
		throws SimilarityException
	{


        this.rel.setAllowSkippedWords(true);
        rel.setMaxParses(3);
        rel.setMaxParseSeconds(60);

        EntityMaintainer em = null;

        String sentence = "This a little bit nice.";

        //RelexInfo ri =
        relex.Sentence ri = rel.processSentence(sentence,em);



        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();

        int sum1 = 0;
        int sum2 = 0;


        for (char c: first_string.toCharArray())
            sum1 += c;

        for (char c: second_string.toCharArray())
            sum2 += c;

        int diff = Math.abs(sum2 - sum1);


        return diff;  //To change body of implemented methods use File | Settings | File Templates.
		// Your similarity computation goes here.

	}

}