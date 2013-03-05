package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;
import relex.RelationExtractor;
import relex.Sentence;
import relex.entity.EntityMaintainer;
import relex.frame.Frame;
import relex.output.SimpleView;

import java.util.Collection;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class RelationSimilarityMeasureOneConstituent
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;
    private RelationExtractor rel;

	public RelationSimilarityMeasureOneConstituent(int n, RelationExtractor rel)
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

        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();


        Sentence ri = rel.processSentence(first_string,em);
        Sentence secsen = rel.processSentence(second_string,em);

        if (ri.getParses().size() == 0 || secsen.getParses().size() == 0)
            return 0;

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

        for (String f : fout) {
            Matcher matcher = pattern.matcher(f);
            matcher.find();
            String f_cut_out = f.substring(matcher.start(),matcher.end());

            for (String g : secfout) {
                Matcher g_matcher = pattern.matcher(g);
                g_matcher.find();

                String g_cut_out = g.substring(g_matcher.start(),g_matcher.end());

                if (f_cut_out.equals(g_cut_out)) {
                    matches++;
                }
            }

        }

        secfout.getClass();

        int sum1 = 0;
        int sum2 = 0;


        Double diff = (matches / fout.length );
        diff = Math.floor(diff*1000) / 1000;

        return diff;  //To change body of implemented methods use File | Settings | File Templates.
		// Your similarity computation goes here.

	}

}