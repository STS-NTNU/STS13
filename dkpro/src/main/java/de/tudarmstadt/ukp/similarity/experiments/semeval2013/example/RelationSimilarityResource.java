package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.dkpro.resource.TextSimilarityResourceBase;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.ResourceSpecifier;
import org.uimafit.descriptor.ConfigurationParameter;
import relex.RelationExtractor;

import java.util.Map;


public class RelationSimilarityResource
	extends TextSimilarityResourceBase
{
	public static final String PARAM_N = "N";
	@ConfigurationParameter(name=PARAM_N, mandatory=false)
	private int n;

    public RelationExtractor rel;

	@SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    public boolean initialize(ResourceSpecifier specifier, Map additionalParams)
        throws ResourceInitializationException
    {
        if (!super.initialize(specifier, additionalParams)) {
            return false;
        }


        //RelationExtractor rel = new RelationExtractor(false);

        this.rel = new RelationExtractor(false);

        this.mode = TextSimilarityResourceMode.text;
        
		measure = new RelationSimilarityMeasure(n, rel);
        
        return true;
    }
}
