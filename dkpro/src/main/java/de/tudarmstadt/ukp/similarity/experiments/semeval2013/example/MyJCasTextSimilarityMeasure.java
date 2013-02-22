package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.algorithms.api.JCasTextSimilarityMeasureBase;
import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;

public class MyJCasTextSimilarityMeasure
        extends JCasTextSimilarityMeasureBase
{
    @SuppressWarnings("unused")
    private int n;

    public MyJCasTextSimilarityMeasure(int n)
    {
        // The configuration parameter is not used right now and intended for illustration purposes only.
        this.n = n;
    }


    @Override
    public double getSimilarity(JCas jcas1, JCas jcas2) throws SimilarityException {
        return 1;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public double getSimilarity(JCas jcas1, JCas jcas2, Annotation coveringAnnotation1, Annotation coveringAnnotation2) throws SimilarityException {
        return 1;  //To change body of implemented methods use File | Settings | File Templates.
    }
}