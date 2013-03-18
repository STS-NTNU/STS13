#!/usr/bin/env python

"""
template file for NTNU submissions that generates a zipfile according to STS13 guidelines
"""

from os import mkdir
from os.path import exists
from subprocess import call

from sklearn.linear_model import LinearRegression

from sts.io import read_system_input, write_scores
from sts.sts13 import test_input_fnames
from ntnu.sts12 import read_train_data, train_ids
from ntnu.sts13 import read_test_data
from ntnu.io import postprocess
from ntnu.feats import takelab_feats


GROUP = "NTNU"

APPROACH = "METHOD1"

DESCRIPTION = \
"""
linear regression on Takelab features 
"""



# pairing of 2012 training data and 2013 test data
id_pairs = [ 
    ("MSRvid", "headlines"),
    ("SMTeuroparl", "SMT"),
    (train_ids, "FNWN"),
    (train_ids, "OnWN") ]

# features to be used
feats = takelab_feats

# learning algorithm
regressor = LinearRegression()

out_dir = "STScore-{}-{}".format(GROUP, APPROACH)
if not exists(out_dir): mkdir(out_dir)

filenames = []

for train_id, test_id in id_pairs:
    X_train, y_train = read_train_data(train_id, feats)
    regressor.fit(X_train, y_train)
    
    X_test = read_test_data(test_id, feats)
    y_test = regressor.predict(X_test)
    
    test_input = read_system_input(test_input_fnames[test_id])
    postprocess(test_input,  y_test)
    
    fname =  "{}/STScore.output.{}.txt".format(out_dir, test_id)
    write_scores(fname, y_test)
    filenames.append(fname)
    
    
descr_fname = "{}/STScore-{}-{}.description.txt".format(out_dir, GROUP, APPROACH)
open(descr_fname, "w").write(DESCRIPTION)
filenames.append(descr_fname)

filenames = " ".join(filenames)

zipfile = "STScore-{}-{}.zip".format(GROUP, APPROACH)

call("zip -rv {} {}".format(zipfile, filenames), 
     shell=True)    
    
    
    
