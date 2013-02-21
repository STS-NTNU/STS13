package de.tudarmstadt.ukp.similarity.experiments.semeval2013.example;

import de.tudarmstadt.ukp.similarity.dkpro.resource.TextSimilarityResourceBase;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.ResourceSpecifier;
import org.uimafit.descriptor.ConfigurationParameter;

import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: stinky
 * Date: 21/2/13
 * Time: 3:25 PM
 * To change this template use File | Settings | File Templates.
 */
public class SentLenSimilarityResource extends TextSimilarityResourceBase {
    public static final String PARAM_LOG = "log";
    @ConfigurationParameter(name=PARAM_LOG, mandatory=false)
    private boolean log;

    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    public boolean initialize(ResourceSpecifier specifier, Map additionalParams)
            throws ResourceInitializationException
    {
        if (!super.initialize(specifier, additionalParams)) {
            return false;
        }

        this.mode = TextSimilarityResourceMode.text;

        measure = new SentLenSimilarityMeasure(log);

        return true;
    }
}
