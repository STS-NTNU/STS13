package de.tudarmstadt.ukp.similarity.experiments.semeval2013;

import de.tudarmstadt.ukp.similarity.experiments.semeval2013.util.Evaluator;
import de.tudarmstadt.ukp.similarity.experiments.semeval2013.util.Features2Arff;
import org.apache.commons.cli.*;

import static de.tudarmstadt.ukp.similarity.experiments.semeval2013.Pipeline.Dataset.*;
import static de.tudarmstadt.ukp.similarity.experiments.semeval2013.Pipeline.EvaluationMetric.PearsonAll;
import static de.tudarmstadt.ukp.similarity.experiments.semeval2013.Pipeline.EvaluationMetric.PearsonMean;
import static de.tudarmstadt.ukp.similarity.experiments.semeval2013.Pipeline.Mode.TEST;
import static de.tudarmstadt.ukp.similarity.experiments.semeval2013.Pipeline.Mode.TRAIN;


public class Pipeline
{
	public enum Mode
	{
		TRAIN,
		TEST
	}
	
	public enum Dataset
	{
		ALL,
		MSRpar,
		MSRvid,
		OnWN,
		SMTeuroparl,
		SMTnews
	}
	
	public enum EvaluationMetric
	{
		PearsonAll,
		PearsonMean
	}
	
	public static final String DATASET_DIR = "file:datasets/semeval-2012"; //"classpath:/datasets/semeval-2012";
	public static final String GOLDSTANDARD_DIR = "file:goldstandards/semeval-2012"; //"classpath:/goldstandards/semeval-2012";
	
	public static final String FEATURES_DIR = "target/features";
	public static final String MODELS_DIR = "target/models";
	public static final String UTILS_DIR = "target/utils";
	public static final String OUTPUT_DIR = "target/output";

    public static final String FILTER = "nSimilarityMeasure.txt";

	public static void main(String[] args)
		throws Exception
	{
		Options options = new Options();
		options.addOption("D", "train", false, "run train mode");
		options.addOption("T", "test", false, "run test mode incl. evaluation (post-submission scenario)");
		options.addOption("S", "submission", false, "run test mode without evaluation (submission scenario)");
		
		CommandLineParser parser = new PosixParser();
		try {
			CommandLine cmd = parser.parse( options, args);
			
			if (cmd.hasOption("D"))
			{
				System.out.println("*** TRAIN MODE ***");
				runTrain();
			}
			if (cmd.hasOption("T"))
			{
				System.out.println("*** TEST MODE (incl. evaluation) ***");
				runTest(true);
			}
			if (cmd.hasOption("S"))
			{
				System.out.println("*** TEST MODE (without evaluation) ***");
				runTest(false);
			}
			
			if (cmd.getOptions().length == 0)
				new HelpFormatter().printHelp(Pipeline.class.getSimpleName(), options, true);
		}
		catch (ParseException e) {
			new HelpFormatter().printHelp(Pipeline.class.getSimpleName(), options);
		}
	}
	
	public static void runTrain()
		throws Exception
	{
		// Generate the features for training data
		FeatureGeneration.generateFeatures(MSRpar, TRAIN);
		FeatureGeneration.generateFeatures(MSRvid, TRAIN);
		FeatureGeneration.generateFeatures(SMTeuroparl, TRAIN);
		
		// Packages features in arff files
		Features2Arff.toArffFileFilter(TRAIN, FILTER, MSRpar, MSRvid, SMTeuroparl);

		// Run the classifier
		Evaluator.runLinearRegressionCV(TRAIN, MSRpar, MSRvid, SMTeuroparl);
		
		// Evaluate
		Evaluator.runEvaluationMetric(TRAIN, PearsonAll, MSRpar, MSRvid, SMTeuroparl);
		Evaluator.runEvaluationMetric(TRAIN, PearsonMean, MSRpar, MSRvid, SMTeuroparl);
	}

	public static void runTest(boolean runEvaluation)
		throws Exception
	{

		// Generate the features for training data
		FeatureGeneration.generateFeatures(MSRpar, TRAIN);
		FeatureGeneration.generateFeatures(MSRvid, TRAIN);
		FeatureGeneration.generateFeatures(SMTeuroparl, TRAIN);
		
		// Generate the features for test data
		FeatureGeneration.generateFeatures(MSRpar, TEST);
		FeatureGeneration.generateFeatures(MSRvid, TEST);
		FeatureGeneration.generateFeatures(SMTeuroparl, TEST);
		FeatureGeneration.generateFeatures(OnWN, TEST);
		FeatureGeneration.generateFeatures(SMTnews, TEST);
		
		// Concatenate all training data
		FeatureGeneration.combineFeatureSets(TRAIN, ALL, MSRpar, MSRvid, SMTeuroparl);
		
		// Package features in arff files
		Features2Arff.toArffFileFilter(TRAIN, FILTER, ALL);
		Features2Arff.toArffFileFilter(TEST, FILTER, MSRpar, MSRvid, SMTeuroparl, OnWN, SMTnews);

		// Run the classifer
		Evaluator.runLinearRegression(ALL, MSRpar, MSRvid, SMTeuroparl, OnWN, SMTnews);
		
		// Evaluate
		// Note: For submission scenario, comment the lines below, as 
		// there is no gold standard present then to compare with.
		if (runEvaluation)
		{
			Evaluator.runEvaluationMetric(TEST, PearsonAll, MSRpar, MSRvid, SMTeuroparl, OnWN, SMTnews);
			Evaluator.runEvaluationMetric(TEST, PearsonMean, MSRpar, MSRvid, SMTeuroparl, OnWN, SMTnews);
		}
	}
}
