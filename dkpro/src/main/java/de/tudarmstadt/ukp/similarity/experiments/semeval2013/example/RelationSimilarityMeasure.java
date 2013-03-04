package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;

import java.util.Collection;

import relex.RelationExtractor;
import relex.Sentence;
import relex.entity.EntityMaintainer;
import relex.frame.Frame;
import relex.output.SimpleView;


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

        /*
            This measure will return the full overlap, the proportion of frames found in both
            sentences
        */

        this.rel.setAllowSkippedWords(true);
        this.rel.setMaxParses(3);
        this.rel.setMaxParseSeconds(60);

        EntityMaintainer em = null;

        String sentence = "Without doubt, it was better to have no agreement at all than a poor one and it is true that, in this instance, the American proposal was a third-rate proposal, and one which was completely unacceptable to Europe.";

        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();

        String [] foo = {"-n 4 -l -t -f -r -a -s Alice ate the mushroom."};

        //rel.main(foo);
        //RelationExtractor.main(foo);

        Sentence ri = rel.processSentence(first_string,em);

        // Need to get out of here if there are no parses
        if (ri.getParses().size() == 0)
            return 0;

        Frame frame = null;
        frame = new Frame();
		String fin = SimpleView.printRelationsAlt(ri.getParses().get(0));
		String[] fout = frame.process(fin);




        Sentence secsen = rel.processSentence(second_string,em);
        Frame secframe = null;
        secframe = new Frame();
        String secfin = SimpleView.printRelationsAlt(secsen.getParses().get(0));
        String[] secfout = secframe.process(secfin);

        float matches = 0;

        for (String f : fout) {

            for (String g : secfout) {

                if (g.contentEquals(f)) {
                    matches++;
                    //System.out.print(f + " \n");
                }
            }

        }

        secfout.getClass();

        int sum1 = 0;
        int sum2 = 0;


        float diff = (matches / fout.length );


        return diff;  //To change body of implemented methods use File | Settings | File Templates.
		// Your similarity computation goes here.

	}

}