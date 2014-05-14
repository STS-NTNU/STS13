import os
from subprocess import call

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR

from ntnu.feats import takelab_lsa_feats, takelab_feats, subsem_best_feats
from ntnu.io import postprocess
import ntnu.sts12 as ntnu_sts12
import ntnu.sts13 as ntnu_sts13
from sts import sts13
from sts.io import read_system_input, write_scores
from sts.sts14 import read_blind_test_data, test_input_fnames


GROUP = "NTNU"

APPROACH = "RUN2"

DESCRIPTION = \
    """
    = TEAM =

    NTNU, Norwegian University of Technology and Science
    emarsi@idi.ntnu.no, andrely@idi.ntnu.no, parthapakray@gmail.com

    STS-en-NTNU-RUN2.zip


    = DESCRIPTION OF RUN =

    Baseline features from previous submission with textual distance features
    based on sublexical semantics representations and bagged SVM RBF kernel regressor.

    = TOOLS USED =

    Takelab feature generation scripts and models.
    Sublexical semantics model trained with Gensim and wcluster.
    SVM regression and bagging using Scikit-Learn.

    = RESOURCES USED =

    Takelab word frequencies and LSA models.
    Subset of English Wikipedia (first 12.5 mill. and 126 mill. words).
    STS12 and STS13 datasets.

    = METHOD TO COMBINE TOOLS AND RESOURCES =

    Machine learning with SVM regression using textual distance metrics as predictive variables.
    Bagging of SVM regressors.


    = COMMENTS =

    """

# pairing of 2012 training and test data to 2014 test data
id_pairs = [
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, "OnWN"),
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, 'deft-forum'),
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, 'deft-news'),
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, 'headlines'),
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, 'images'),
    (ntnu_sts12.train_ids, ntnu_sts12.test_ids, ntnu_sts13.test_ids, 'tweet-news')]

# features to be used
feats = takelab_feats + takelab_lsa_feats + subsem_best_feats

# learning algorithm in default setting
regressor = BaggingRegressor(SVR(), verbose=1, n_jobs=3, n_estimators=100, max_features=0.8, max_samples=0.8)


out_dir = "STS-en-{}-{}".format(GROUP, APPROACH)
if not os.path.exists(out_dir): os.mkdir(out_dir)

filenames = []

for sts12_train_id, sts12_test_id, sts13_test_id, sts14_test_id in id_pairs:
    # combine 2012, 2013 training and test data
    X_sts12_train, y_sts12_train = ntnu_sts12.read_train_data(sts12_train_id, feats)
    X_sts12_test, y_sts12_test = ntnu_sts12.read_test_data(sts12_test_id, feats)
    X_sts13_test, y_sts13_test = sts13.read_test_data(sts13_test_id, feats)
    X_train = np.vstack([X_sts12_train, X_sts12_test, X_sts13_test])
    y_train = np.hstack([y_sts12_train, y_sts12_test, y_sts13_test])

    regressor.fit(X_train, y_train)

    X_test = read_blind_test_data(sts14_test_id, feats)
    y_test = regressor.predict(X_test)

    test_input = read_system_input(test_input_fnames[sts14_test_id])
    postprocess(test_input,  y_test)

    fname =  "{}/STS-en.output.{}.txt".format(out_dir, sts14_test_id)
    write_scores(fname, y_test)
    filenames.append(fname)

descr_fname = "{}/STS-en-{}-{}.description.txt".format(out_dir, GROUP, APPROACH)
open(descr_fname, "w").write(DESCRIPTION)
filenames.append(descr_fname)

filenames = " ".join(filenames)

zipfile = "STS-en-{}-{}.zip".format(GROUP, APPROACH)

call("zip -rv {} {}".format(zipfile, filenames),
     shell=True)

