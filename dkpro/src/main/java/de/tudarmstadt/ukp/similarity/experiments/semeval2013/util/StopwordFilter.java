package de.tudarmstadt.ukp.similarity.experiments.semeval2013.util;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.uimafit.component.JCasAnnotator_ImplBase;
import org.uimafit.descriptor.ConfigurationParameter;
import org.uimafit.util.JCasUtil;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


public class StopwordFilter
	extends JCasAnnotator_ImplBase
{
	public static final String PARAM_STOPWORD_LIST = "StopwordList";
	@ConfigurationParameter(name=PARAM_STOPWORD_LIST, mandatory=true)
	private String stopwordList;

	public static final String PARAM_ANNOTATION_TYPE_NAME = "AnnotationType";
	@ConfigurationParameter(name=PARAM_ANNOTATION_TYPE_NAME, defaultValue="de.tudarmstadt.ukp.dkpro.core.type.Token")
	private String annotationTypeName;

	public static final String PARAM_STRING_REPRESENTATION_METHOD_NAME = "StringRepresentationMethodName";
	@ConfigurationParameter(name=PARAM_STRING_REPRESENTATION_METHOD_NAME, defaultValue="getCoveredText")
	private String stringRepresentationMethodName;

	private Class<? extends Annotation> annotationType;
	private Set<String> stopwords;

	@SuppressWarnings("unchecked")
	@Override
	public void initialize(UimaContext context)
		throws ResourceInitializationException
	{
		super.initialize(context);

		try {
			annotationType = (Class<? extends Annotation>) Class.forName(annotationTypeName);
		}
		catch (ClassNotFoundException e) {
			throw new ResourceInitializationException(e);
		}

		PathMatchingResourcePatternResolver r = new PathMatchingResourcePatternResolver();
        stopwordList = "file:stopwords/stopwords_english_punctuation.txt";
        Resource res = r.getResource(stopwordList);

        File f;
		try {
			f = res.getFile();
		}
		catch (IOException e) {
			throw new ResourceInitializationException(e);
		}
		
        try {
			loadStopwords(new FileInputStream(f));
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
    }

	@SuppressWarnings("deprecation")
	@Override
	public void process(JCas jcas)
		throws AnalysisEngineProcessException
	{
		List<Annotation> itemsToRemove = new ArrayList<Annotation>();

    	// Check all annotations if they are stopwords
		for (Annotation annotation : JCasUtil.iterate(jcas, annotationType))
		{
			try {
				String word = (String) annotation.getClass().getMethod(stringRepresentationMethodName, new Class[]{}).invoke(annotation, new Object[]{});

				if (isStopword(word)) {
	            	itemsToRemove.add(annotation);
	            }
			}
			catch (IllegalArgumentException e) {
				throw new AnalysisEngineProcessException(e);
			}
			catch (SecurityException e) {
				throw new AnalysisEngineProcessException(e);
			}
			catch (IllegalAccessException e) {
				throw new AnalysisEngineProcessException(e);
			}
			catch (InvocationTargetException e) {
				throw new AnalysisEngineProcessException(e);
			}
			catch (NoSuchMethodException e) {
				throw new AnalysisEngineProcessException(e);
			}
		}

    	// Remove all stopwords from index
    	for (Annotation a : itemsToRemove)
    	{
            a.removeFromIndexes();
        }
    }

	private boolean isStopword(String item)
	{
		assert item.length() > 0;

	    // item is in the stopword list
		try
		{
			if (stopwords.contains(item.toLowerCase())) {
				return true;
			}
		}
		catch (NullPointerException e)
		{
			// Ignore this token
			return true;
		}

	    // does not start with a letter
	    if (!firstCharacterIsLetter(item)) {
	        return true;
	    }

        return false;
	}

	private boolean firstCharacterIsLetter(String item)
	{
		if (item.matches("^[A-Za-z].+")) {
            return true;
        }
        else {
            return false;
        }
	}

	private void loadStopwords(InputStream is)
		throws ResourceInitializationException
	{
		stopwords = new HashSet<String>();

		try {
			String line;
			BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
			while ((line = br.readLine()) != null) {
				stopwords.add(line);
			}
		}
		catch (IOException e) {
			throw new ResourceInitializationException(e);
		}
    }
}
