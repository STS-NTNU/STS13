#!/usr/bin/env python

"""
Simple example of using your own feature
"""

from sts.io import read_system_input
from sts.score import correlation
from sts.sts12 import test_input_fnames
from ntnu.sts12 import train_ids, read_train_data, read_test_data
from ntnu.io import postprocess

from sklearn.svm import SVR


# list of tuples (train_ids, test_ids) defining training and test data sets;
# in this case train and test on STS12 MSRpar data only
id_pairs = [ 
    ("MSRpar", "MSRpar") ]

# features to be used
feats = ["my_feat"]

# use support vector reressor in default setting
regressor = SVR()


for train_id, test_id in id_pairs:
    # create training data and labels
    train_feat, train_scores = read_train_data(train_id, feats)
    
    # train regressor
    regressor.fit(train_feat, train_scores)
    
    # create test data and labels
    test_feat, test_scores = read_test_data(test_id, feats)
    
    # apply regressor to test data 
    sys_scores = regressor.predict(test_feat)
    
    # postprocess
    sys_input = read_system_input(test_input_fnames[test_id])
    postprocess(sys_input,  sys_scores)
    
    if isinstance(train_id, tuple):
        train_id = " + ".join(train_id)
    
    # compute correlation score
    r = correlation(sys_scores, test_scores)
    
    print "{:32s} {:32s} {:2.4f}".format(train_id, test_id, r)
        
