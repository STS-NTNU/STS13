package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;

import java.util.Collection;

/**
 * Created with IntelliJ IDEA.
 * User: stinky
 * Date: 21/2/13
 * Time: 3:19 PM
 * To change this template use File | Settings | File Templates.
 */
public class SentLenSimilarityMeasure extends TextSimilarityMeasureBase {
    private boolean log = false;

    public SentLenSimilarityMeasure(boolean log) {
        this.log = log;
    }

    @Override
    public double getSimilarity(Collection<String> strings, Collection<String> strings2) throws SimilarityException {
        int l1 = strings.iterator().next().split(" ").length;
        int l2 = strings2.iterator().next().split(" ").length;

        int diff = Math.abs(l1 - l2);

        if (log && (diff != 0)) {
            return Math.log(diff);
        }

        return diff;  //To change body of implemented methods use File | Settings | File Templates.
    }
}
