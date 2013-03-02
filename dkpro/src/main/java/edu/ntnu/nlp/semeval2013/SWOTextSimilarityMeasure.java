package edu.ntnu.nlp.semeval2013;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;
import edu.ntnu.nlp.sim.text.SemanticWordOrderSim;
import edu.ntnu.nlp.sim.text.TextSim;
import edu.ntnu.nlp.sim.word.WNSim;
import edu.ntnu.nlp.sim.word.WordSim;

import java.util.Collection;

public class SWOTextSimilarityMeasure
        extends TextSimilarityMeasureBase {
    TextSim textSim;

    public SWOTextSimilarityMeasure() {

        WordSim wordSim = WNSim.LeacockAndChodorow();
        textSim = new SemanticWordOrderSim(wordSim, 0.8);
    }

    @Override
    public double getSimilarity(Collection<String> stringList1,
                                Collection<String> stringList2)
            throws SimilarityException
    {
        String string1 = stringList1.iterator().next();
        String string2 = stringList2.iterator().next();
        return textSim.getSim(string1, string2);
    }
}
