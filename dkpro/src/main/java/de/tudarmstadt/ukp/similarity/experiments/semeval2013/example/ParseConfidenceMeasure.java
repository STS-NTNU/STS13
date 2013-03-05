package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;
import relex.RelationExtractor;
import relex.Sentence;
import relex.entity.EntityMaintainer;
import relex.frame.Frame;
import relex.output.SimpleView;

import java.util.Collection;


public class ParseConfidenceMeasure
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;
    private RelationExtractor rel;

	public ParseConfidenceMeasure(int n, RelationExtractor rel)
	{                                                                          		// The configuration parameter is not used right now and intended for illustration purposes only.
		this.n = n;
        this.rel = rel;
	}

    public double getSimilarity(Collection<String> strings,
			Collection<String> strings2)
		throws SimilarityException
	{

        /*
            This measure will return the full overlap, the proportion of frames found in both
            sentences
        */

        this.rel.setAllowSkippedWords(true);
        this.rel.setMaxParses(1);
        this.rel.setMaxParseSeconds(60);

        EntityMaintainer em = null;

        String sentence = "Thus, it is urgent that the staff of the interservice group are very quickly strengthened at the heart of the Secretary-General of the Commission, so that all the proposed act of general scope can be accompanied, during their examination by the college on the basis of Article 299 (2) of a fiche d'impact detailed.";
        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();

        String [] foo = {"-n 4 -l -t -f -r -a -s Alice ate the mushroom."};

        // System.out.println(first_string + "\n");

        Sentence ri = rel.processSentence(first_string,em);
        Sentence secsen = rel.processSentence(second_string,em);

        // Need to get out of here if there are no parses
        if (ri.getParses().size() == 0 || secsen.getParses().size() == 0)
            return 0.0;

        Double first_confidence = ri.getParses().get(0).getTruthValue().getConfidence();
        Double sec_confidence = secsen.getParses().get(0).getTruthValue().getConfidence();


        Double diff = Math.abs(first_confidence-sec_confidence);

        diff = Math.floor(diff*1000) / 1000;

        return diff;  //To change body of implemented methods use File | Settings | File Templates.
		// Your similarity computation goes here.

	}

}