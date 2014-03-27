import os
from subprocess import call

import numpy as np

from sklearn.svm import SVR

from ntnu.feats import takelab_lsa_feats, takelab_feats
from ntnu.io import postprocess

from ntnu.sts12 import read_train_data, train_ids, read_test_data, test_ids
from sts.io import read_system_input, write_scores
from sts.sts14 import read_blind_test_data, test_input_fnames


GROUP = "NTNU"

APPROACH = "RUN1"

DESCRIPTION = \
    """
    = TEAM =

    NTNU

    = DESCRIPTION OF RUN =

    Baseline features from previous submission with textual distance features
    based on sublexical semantics representations.

    = TOOLS USED =

    Takelab feature generation.
    Sublexical semantics model trained with Gensim.
    SVM regression using Scikit-Learn.

    = RESOURCES USED =

    * Monolingual corpora

    Subset of English Wikipedia.

    = METHOD TO COMBINE TOOLS AND RESOURCES =

    Machine learning with SVM regression using textual distance metrics as predictive variables.


    = COMMENTS =

    """

# pairing of 2012 training and test data to 2014 test data
id_pairs = [
    (train_ids, test_ids, "OnWN"),
    (train_ids, test_ids, 'deft-forum'),
    (train_ids, test_ids, 'deft-news'),
    (train_ids, test_ids, 'headlines'),
    (train_ids, test_ids, 'images'),
    (train_ids, test_ids, 'tweet-news')]

# features to be used
feats = takelab_feats + takelab_lsa_feats + ['subsem-lsitfidf-wiki8-n4-c2000-min5-raw-cos'] # dkpro_feats + gleb_feats

# learning algorithm in default setting
regressor = SVR()

out_dir = "STScore-{}-{}-2014".format(GROUP, APPROACH)
if not os.path.exists(out_dir): os.mkdir(out_dir)

filenames = []

for sts12_train_id, sts12_test_id, sts14_test_id in id_pairs:
    # combine 2012 training and test data
    X_sts12_train, y_sts12_train = read_train_data(sts12_train_id, feats)
    X_sts12_test, y_sts12_test = read_test_data(sts12_test_id, feats)
    X_train = np.vstack([X_sts12_train, X_sts12_test])
    y_train = np.hstack([y_sts12_train, y_sts12_test])

    regressor.fit(X_train, y_train)

    X_test = read_blind_test_data(sts14_test_id, feats)
    y_test = regressor.predict(X_test)

    test_input = read_system_input(test_input_fnames[sts14_test_id])
    postprocess(test_input,  y_test)

    fname =  "{}/STScore.output.{}.txt".format(out_dir, sts14_test_id)
    write_scores(fname, y_test)
    filenames.append(fname)

descr_fname = "{}/STScore-{}-{}.description.txt".format(out_dir, GROUP, APPROACH)
open(descr_fname, "w").write(DESCRIPTION)
filenames.append(descr_fname)

filenames = " ".join(filenames)

zipfile = "STScore-{}-{}.zip".format(GROUP, APPROACH)

call("zip -rv {} {}".format(zipfile, filenames),
     shell=True)

