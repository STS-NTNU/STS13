#!/usr/bin/env python

"""
template file for NTNU submissions that generates a zipfile according to STS13 guidelines


Downloading test data and uploading runs
-------------------------------------------------------

The download test directory (/home/sts) contains two files:

 /home/sts/test.tgz        The test files for the core STS task
 /home/sts/test-typed.tgz  The test files fof the typed STS task

Participant teams will be allowed to submit three runs at most for
each task. Each run will be delivered as a compressed file (zip or
gzip format) and will contain:

- a directory containing:
 - the answer files for all datasets
 - a description file (see below)

The participants are required to follow the naming convention, as
follows:

 STScore-GROUP-APPROACH.zip

or

 STStyped-GROUP-APPROACH.zip

And should contain respectively:

 STScore-GROUP-APPROACH/
    STScore-GROUP-APPROACH.description.txt
    STScore.output.FNWN.txt
    STScore.output.OnWN.txt
    STScore.output.SMT.txt
    STScore.output.headlines.txt

or

 STStyped-GROUP-APPROACH/
    STStyped-GROUP-APPROACH.description.txt
    STStyped.output.europeana.txt

where GROUP identifies which group made the submission (please
use the same name as in the registration).

and APPROACH identifies each run.

Each run needs to be accompanied by a text file describing the method,
tools and resources used in the run. This file will help the
organizers produce an informative report on the task. Please fill the
required information following the format in the following files as
made available with the test data:

 STScore-GROUP-APPROACH.description.txt
 STStyped-GROUP-APPROACH.description.txt

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
    
    
    
