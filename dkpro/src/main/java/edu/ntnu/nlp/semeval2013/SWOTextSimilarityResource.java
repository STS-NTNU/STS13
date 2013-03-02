package edu.ntnu.nlp.semeval2013;

import de.tudarmstadt.ukp.similarity.dkpro.resource.TextSimilarityResourceBase;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.ResourceSpecifier;

import java.util.Map;


public class SWOTextSimilarityResource
	extends TextSimilarityResourceBase
{
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    public boolean initialize(ResourceSpecifier specifier, Map additionalParams)
            throws ResourceInitializationException
    {
        if (!super.initialize(specifier, additionalParams)) {
            return false;
        }

        this.mode = TextSimilarityResourceMode.text;
        measure = new SWOTextSimilarityMeasure();
        return true;
    }
}
