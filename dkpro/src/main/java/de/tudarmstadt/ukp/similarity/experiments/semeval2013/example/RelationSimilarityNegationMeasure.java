package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;
import relex.RelationExtractor;
import relex.Sentence;
import relex.entity.EntityMaintainer;
import relex.frame.Frame;
import relex.output.SimpleView;

import java.util.Collection;
import java.util.regex.Pattern;


public class RelationSimilarityNegationMeasure
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;
    private RelationExtractor rel;

	public RelationSimilarityNegationMeasure(int n, RelationExtractor rel)
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

        String sentence = "Thus, I don't like that";

        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();

        Sentence ri = rel.processSentence(first_string,em);
        Sentence secsen = rel.processSentence(second_string,em);

        // Need to get out of here if there are no parses
        if (ri.getParses().size() == 0 || secsen.getParses().size() == 0)
            return 0.0;

        Frame frame = null;
        frame = new Frame();
		String fin = SimpleView.printRelationsAlt(ri.getParses().get(0));
		String[] fout = frame.process(fin);


        Frame secframe = null;
        secframe = new Frame();
        String secfin = SimpleView.printRelationsAlt(secsen.getParses().get(0));
        String[] secfout = secframe.process(secfin);

        if (fout.length == 0 || secfout.length == 0) {
            return 0.0;
        }

        Pattern pattern = Pattern.compile("[^,]+");

        Double matches = 0.0;

        boolean negated_f = false;
        boolean negated_g = false;

        for (String f : fout) {

            if (f.contains("Negation")) {
                negated_f = true;
            }
        }

        for (String g : secfout) {

            if (g.contains("Negation")) {
                negated_g = true;
            }
        }

        if (negated_f != negated_g)
            return 1;
        else
            return 0;


	}

}