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

import numpy as np

from sklearn.svm import SVR

from sts.io import read_system_input, write_scores
from sts.sts13 import test_input_fnames
from ntnu.sts12 import read_train_data, train_ids, read_test_data, test_ids
from ntnu.sts13 import read_blind_test_data
from ntnu.io import postprocess
from ntnu.feats import all_feats


GROUP = "NTNU"

APPROACH = "RUN1"


# Please include a description of your submission, following the format
# below. 
# Please don't modify lines starting with = unless instructed to do so.
# Please delete lines starting with [. All lines starting with [ will be
# ignored.

DESCRIPTION = \
"""
= TEAM =

[Please include affiliation and name of first author]

= DESCRIPTION OF RUN =

[Please include a short paragraph describing your method for the run]

= TOOLS USED =

[Please keep those tools that you used, delete those you did not use, and add more lines if needed]

* Part of speech tagger
* Lemmatizer
* Multiword expressions recognition
* Syntax parser
* Semantic Role Labeling
* Word Sense Disambiguation
* Lexical Substitution
* Distributional similarity
* Knowledge-based similarity
* Time and date resolution
* Named Entity recognition
* Sentiment Analysis
* Metaphor and/or Metonymy
* Logical Inference
* Textual Entailment
* Correference
* Scoping
* ... (add as needed)

= RESOURCES USED =

[Please keep those resources that you used, delete those you did not use, and add more lines if needed]

* Monolingual corpora
* Multilingual corpora
* Tables of paraphrases
* WordNet
* PropBank
* FrameNet
* Ontonotes
* Repositories for named-entities and acronyms
* Other dictionaries (please specify)
* Wikipedia
* VerbOcean
* Dirt
* Lin's thesaurus
* Other distributional similarity thesauri (please specify)
* ... (add as needed)

= METHOD TO COMBINE TOOLS AND RESOURCES =

[Please summarize the method to combine the tools and resources, mentioning whether it's heuristic, or uses machine learning, etc.]


= COMMENTS =

[Please include any comment you might have to improve the task in the future]

"""


# pairing of 2012 training and test data to 2013 test data
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


# learning algorithm in default setting
regressor = SVR()

out_dir = "STScore-{}-{}".format(GROUP, APPROACH)
if not exists(out_dir): mkdir(out_dir)

filenames = []

for sts12_train_id, sts12_test_id, sts13_test_id in id_pairs:
    # combine 2012 training and test data 
    X_sts12_train, y_sts12_train = read_train_data(sts12_train_id, feats)
    X_sts12_test, y_sts12_test = read_test_data(sts12_test_id, feats)
    X_train = np.vstack([X_sts12_train, X_sts12_test])
    y_train = np.hstack([y_sts12_train, y_sts12_test])
    
    regressor.fit(X_train, y_train)
    
    X_test = read_blind_test_data(sts13_test_id, feats)
    y_test = regressor.predict(X_test)
    
    test_input = read_system_input(test_input_fnames[sts13_test_id])
    postprocess(test_input,  y_test)
    
    fname =  "{}/STScore.output.{}.txt".format(out_dir, sts13_test_id)
    write_scores(fname, y_test)
    filenames.append(fname)
    
    
descr_fname = "{}/STScore-{}-{}.description.txt".format(out_dir, GROUP, APPROACH)
open(descr_fname, "w").write(DESCRIPTION)
filenames.append(descr_fname)

filenames = " ".join(filenames)

zipfile = "STScore-{}-{}.zip".format(GROUP, APPROACH)

call("zip -rv {} {}".format(zipfile, filenames), 
     shell=True)    
    
    
    
