#!/usr/bin/env python

"""
Combination of all features
"""

# make sure have numpy installed 

import numpy as np

# make sure you have lib/python in your PYTHONPATH, e.g. using
#  export PYTHONPATH=$PYTHONPATH:~/Projects/SemTextSim/github/STS13/lib/python

from sts.io import read_system_input
from sts.score import correlation
from sts.sts12 import test_input_fnames
from ntnu.sts12 import train_ids, read_train_data, read_test_data
from ntnu.io import postprocess
from ntnu.feats import all_feats, dkpro_feats, takelab_feats

# make sure you have sklearn installed

from sklearn.svm import SVR


# ids of training and testing data sets
id_pairs = [ 
    ("MSRpar", "MSRpar"),
    ("MSRvid", "MSRvid"),
    ("SMTeuroparl", "SMTeuroparl"),
    ("SMTeuroparl", "surprise.SMTnews"),
    (train_ids, "surprise.OnWN") ]


# features to be used
#feats = dkpro_feats + takelab_feats
feats = all_feats


# learning algorithms, one per test set, where SVR settings result from
# grid-search.sh
#regressors = [
    #SVR(C=50,  epsilon=0.2, gamma=0.02),
    #SVR(C=200, epsilon=0.5, gamma=0.02),
    #SVR(C=100, epsilon=0.2, gamma=0.02),
    #SVR(C=100, epsilon=0.2, gamma=0.02),
    #SVR(C=10,  epsilon=0.5, gamma=0.02)
    #]

regressors = [SVR() for i in range(5)]


for (train_id, test_id), regressor in zip(id_pairs, regressors):
    train_feat, train_scores = read_train_data(train_id, feats)
    regressor.fit(train_feat, train_scores)
    
    test_feat, test_scores = read_test_data(test_id, feats)
    sys_scores = regressor.predict(test_feat)
    
    sys_input = read_system_input(test_input_fnames[test_id])
    postprocess(sys_input,  sys_scores)
    
    if isinstance(train_id, tuple):
        train_id = "+".join(train_id)
    
    print "{:32s} {:32s} {:2.2f}".format(
        train_id, 
        test_id, 
        correlation(sys_scores, test_scores))