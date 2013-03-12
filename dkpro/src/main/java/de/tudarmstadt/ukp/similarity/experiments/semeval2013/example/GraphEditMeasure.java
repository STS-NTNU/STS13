package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;


import de.tudarmstadt.ukp.similarity.algorithms.api.SimilarityException;
import de.tudarmstadt.ukp.similarity.algorithms.api.TextSimilarityMeasureBase;
import no.roek.nlpged.algorithm.GraphEditDistance;
import no.roek.nlpged.application.App;
import no.roek.nlpged.application.Config;
import no.roek.nlpged.graph.Graph;
import no.roek.nlpged.misc.EditWeightService;
import no.roek.nlpged.preprocessing.DependencyParser;
import org.maltparser.core.exception.MaltChainedException;
import relex.RelationExtractor;
import relex.entity.EntityMaintainer;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;


public class GraphEditMeasure
	extends TextSimilarityMeasureBase
{
	@SuppressWarnings("unused")
	private int n;
    private Map<String,Double> deprelEditWeights;
    private Map<String,Double> posEditWeights;
    private DependencyParser depParser;


    public GraphEditMeasure(int n, DependencyParser depParser, Map<String, Double> posEditWeights, Map<String, Double> deprelEditWeights)
	{                                                                          		// The configuration parameter is not used right now and intended for illustration purposes only.
		this.n = n;
        this.deprelEditWeights = deprelEditWeights;
        this.posEditWeights = posEditWeights;
        this.depParser = depParser;
	}

    public double getSimilarity(Collection<String> strings,
			Collection<String> strings2)
		throws SimilarityException
	{

        /*
            This measure will return the full overlap, the proportion of frames found in both
            sentences
        */



        EntityMaintainer em = null;

        String first_string =  strings.iterator().next();
        String second_string =  strings2.iterator().next();


        String [] texts = {first_string, second_string};

        double score = 0.0;


        try {

            Graph g1 = this.depParser.dependencyParse("1", texts[0]);
            Graph g2 = this.depParser.dependencyParse("2", texts[1]);
            GraphEditDistance ged = new GraphEditDistance(g1, g2, posEditWeights, deprelEditWeights);

            score = ged.getDistance();

        }
        catch (MaltChainedException e) {
            System.err.println("Caught MaltChainedException: " + e.getMessage());
            e.printStackTrace();
        }


        return (score);

	}

}