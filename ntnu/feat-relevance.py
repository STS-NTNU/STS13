#!/usr/bin/env python

"""
Example of computing feature relevance with the DKPro system
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
from ntnu.feats import all_feats

# make sure you have sklearn installed

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# ids of training and testing data sets
id_pairs = [ 
    ("MSRpar", "MSRpar"),
    ("MSRvid", "MSRvid"),
    ("SMTeuroparl", "SMTeuroparl"),
    ("SMTeuroparl", "surprise.SMTnews"),
    (train_ids, "surprise.OnWN") ]
    
# features to be used
feats = all_feats

# learning algorithm
regressor = LinearRegression()
# Used with default setting here. However, in the real DKPro system, its setting
# were probably optmized by a CV gridsearch on the training data


# TODO: this approach is brain dead, because it keeps reading features from files

print "{:64s}".format("Features:"),

print " ".join(["{:>16s}".format(p[1]) for p in id_pairs])


for feat in feats:
    print "{:64s}".format(feat),
    
    for train_id, test_id in id_pairs:
        train_feat, train_scores = read_train_data(train_id, [feat])
        regressor.fit(train_feat, train_scores)
        
        test_feat, test_scores = read_test_data(test_id, [feat])
        sys_scores = regressor.predict(test_feat)
        
        sys_input = read_system_input(test_input_fnames[test_id])
        postprocess(sys_input,  sys_scores)
        
        if isinstance(train_id, tuple):
            train_id = "+".join(train_id)
        
        print "{:16.2f}".format(correlation(sys_scores, test_scores)),
    print
    