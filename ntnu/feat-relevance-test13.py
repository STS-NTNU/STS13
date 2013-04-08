#!/usr/bin/env python

"""
Feature relevance on TST13 test data
"""

# make sure have numpy installed 

import numpy as np

# make sure you have lib/python in your PYTHONPATH, e.g. using
#  export PYTHONPATH=$PYTHONPATH:~/Projects/SemTextSim/github/STS13/lib/python

from sts.io import read_system_input
from sts.score import correlation
from sts.sts13 import test_input_fnames
from ntnu.sts12 import train_ids, read_train_data, read_test_data, test_ids
from ntnu import sts13 
from ntnu.io import postprocess
from ntnu.feats import all_feats

# make sure you have sklearn installed

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# pairing of 2012 training and test data to 2013 test data as in NTNU-RUN-1
id_pairs = [ 
    (train_ids,     
     test_ids, 
     "headlines"),
    ("SMTeuroparl", 
     ("SMTeuroparl", "surprise.SMTnews"), 
     "SMT"),
    (train_ids,
     test_ids,
     "FNWN"),
    (train_ids,
     test_ids,
     "OnWN") ]    

# features to be used
feats = all_feats

# learning algorithm
#regressor = SVR()
regressor = LinearRegression()


# TODO: this approach is brain dead, because it keeps reading features from files

print "{:64s}\t".format("Features:"),

print " ".join(["{:>16s}\t".format(p[2]) for p in id_pairs])


for feat in feats:
    print "{:64s}\t".format(feat),
    
    for sts12_train_id, sts12_test_id, sts13_test_id in id_pairs:
        # combine 2012 training and test data 
        X_sts12_train, y_sts12_train = read_train_data(sts12_train_id, [feat])
        X_sts12_test, y_sts12_test = read_test_data(sts12_test_id, [feat])
        X_train = np.vstack([X_sts12_train, X_sts12_test])
        y_train = np.hstack([y_sts12_train, y_sts12_test])
    
        regressor.fit(X_train, y_train)
        
        X_test, y_test = sts13.read_test_data(sts13_test_id, [feat])
        sys_scores = regressor.predict(X_test)
        
        sys_input = read_system_input(test_input_fnames[sts13_test_id])
        postprocess(sys_input,  sys_scores)
        
        print "{:16.2f}\t".format(correlation(sys_scores, y_test)),
    print
    