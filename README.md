STS13
=====

Semantic Textual Similarity task 2012/2013/2014




------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------

* numpy
* sklearn (scikit-learn)
* nltk (may require X11 under OS X)

* Google n-gram word counts and Takelab LSA models under directory _data


------------------------------------------------------------------------------
Python path
------------------------------------------------------------------------------

Make sure you have lib/python in your PYTHONPATH, e.g. in Bash use

$ export PYTHONPATH=$PYTHONPATH:~/Projects/SemTextSim/github/STS13/lib/python



------------------------------------------------------------------------------
Example: generating Takelab features for new STS14 trial data 
------------------------------------------------------------------------------

1. Make sure you have Google n-gram word counts and Takelab LSA models
   under directory _data

2. Add 2014 trial data files under new directory data/STS2014-trial

3. Create lib/python/sts/sts14.py defining dirs, ids and filenames for STS14
   trial data
   
4. Create lib/python/ntnu/sts14.py defining dirs and filenames of features 
   for STS14 trial data

5. Create Takelab features by adding function to ntnu/make-takelab-feat.py:
   
   make_feats(sts.sts14.trial_input_fnames, 
              ntnu.sts14.test_dir,
              with_lsa)
              
   Temporary comment out calls to make_feats for other STS datasets
   
6. Change to dir ./ntnu and run ./make-takelab-feat.py

   New features appear in files out/STS2014-trial/<dataset-id>/<feat-name>.txt
   

------------------------------------------------------------------------------
Example: testing a new feature
------------------------------------------------------------------------------

Suppose we have a new feature called "my_feat" that we want to try on the
MSRpar dataset from STS12.

1. Add the feature to the training and test data as files

   out/STS2012-train/MSRpar/my_feat.txt
   out/STS2012-test/MSRpar/my_feat.txt
   
2. Run a script ntnu/my_feat.py

   Check comments in the script 
   
