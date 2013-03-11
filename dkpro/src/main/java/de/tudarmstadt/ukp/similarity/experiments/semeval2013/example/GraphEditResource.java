package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.dkpro.resource.TextSimilarityResourceBase;
import no.roek.nlpged.application.Config;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.ResourceSpecifier;
import org.uimafit.descriptor.ConfigurationParameter;
import relex.RelationExtractor;

import org.maltparser.MaltParserService;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import no.roek.nlpged.algorithm.GraphEditDistance;
import no.roek.nlpged.graph.Graph;
import no.roek.nlpged.graph.Node;
import no.roek.nlpged.misc.EditWeightService;
import no.roek.nlpged.preprocessing.DependencyParser;

import org.maltparser.core.exception.MaltChainedException;

import com.konstantinosnedas.HungarianAlgorithm;

import java.util.Map;



public class GraphEditResource
	extends TextSimilarityResourceBase
{
	public static final String PARAM_N = "N";
	@ConfigurationParameter(name=PARAM_N, mandatory=false)
	private int n;

    public DependencyParser depParser;
    public Map<String, Double> posEditWeights;
    public Map<String, Double> deprelEditWeights;

	@SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    public boolean initialize(ResourceSpecifier specifier, Map additionalParams)
        throws ResourceInitializationException
    {
        if (!super.initialize(specifier, additionalParams)) {
            return false;
        }



        try {
            Config cs = new Config("app.properties");
            DependencyParser depParser = new DependencyParser(cs.getProperty("MALT_PARAMS"), cs.getProperty("POSTAGGER_PARAMS"));
            Map<String, Double> posEditWeights = EditWeightService.getEditWeights(cs.getProperty("POS_SUB_WEIGHTS"), cs.getProperty("POS_INSDEL_WEIGHTS"));
            Map<String, Double> deprelEditWeights = EditWeightService.getInsDelCosts(cs.getProperty("DEPREL_INSDEL_WEIGHTS"));

            this.depParser = depParser;
            this.posEditWeights = posEditWeights;
            this.deprelEditWeights = deprelEditWeights;

        }
        catch (MaltChainedException e) {
            System.err.println("Caught FileNotFoundException: " + e.getMessage());
        }
        catch (ClassNotFoundException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
        catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }





        this.mode = TextSimilarityResourceMode.text;





		//measure = new GraphEditMeasure(n);
        measure = new GraphEditMeasure(n, depParser, posEditWeights, deprelEditWeights);
        
        return true;
    }
}
