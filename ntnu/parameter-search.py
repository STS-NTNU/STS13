# Attempts at paramter search for STS14
#
# Trains three models:
# 1) Model with default parameters.
# 2) Model with parameters optimized by CV on th training test.
# 3) Model with parameters optimized by evaluating on a validation part of the test set. The first half of the test
#    set is used for parameter search and the second for evaluation.
#
# Based on the ntnu-template.py code.

# Results aren't really that interesting, bur here they are for the Takelab features.

# Large grid results
# headlines	0.6821	0.1183	0.6705	{'epsilon': 0.1, 'C': 0.3, 'gamma': 0.003}	0.6883	0.6656	{'epsilon': 0.1, 'C': 0.3, 'gamma': 0.01}
# SMT	0.3588	0.5140	0.3262	{'epsilon': 0.1, 'C': 3, 'gamma': 0.3}	0.3856	0.4055	{'epsilon': 0.3, 'C': 0.1, 'gamma': 0.03}
# FNWN	0.3460	0.1183	0.3877	{'epsilon': 0.1, 'C': 0.3, 'gamma': 0.003}	0.4439	0.3672	{'epsilon': 1, 'C': 0.0003, 'gamma': 1e-05}
# OnWN	0.4786	0.1183	0.5868	{'epsilon': 0.1, 'C': 0.3, 'gamma': 0.003}	0.6165	0.5767	{'epsilon': 1, 'C': 0.003, 'gamma': 0.1}
# Mean: 0.4664	0.2173	0.4928	0.5336	0.5038

# Grid results
# headlines	0.6821	0.1185	0.6744	{'epsilon': 0.1, 'C': 10, 'gamma': 0.0001}	0.7050	0.6762	{'epsilon': 0.1, 'C': 100, 'gamma': 0.001}
# SMT	0.3588	0.5149	0.3390	{'epsilon': 0.3, 'C': 1, 'gamma': 0.3}	0.3856	0.4055	{'epsilon': 0.3, 'C': 0.1, 'gamma': 0.03}
# FNWN	0.3460	0.1185	0.3822	{'epsilon': 0.1, 'C': 10, 'gamma': 0.0001}	0.4850	0.4055	{'epsilon': 1, 'C': 30, 'gamma': 0.003}
# OnWN	0.4786	0.1185	0.5839	{'epsilon': 0.1, 'C': 10, 'gamma': 0.0001}	0.6165	0.5767	{'epsilon': 1, 'C': 0.003, 'gamma': 0.1}
# Mean: 0.4664	0.2176	0.4949	0.5480	0.5160

# Small grid results
# headlines	0.6821	0.1165	0.6648	{'epsilon': 1, 'C': 0.3, 'gamma': 0.01}	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.01}
# SMT	0.3588	0.4501	0.2555	{'epsilon': 1, 'C': 0.01, 'gamma': 0.1}	0.3856	0.4055	{'epsilon': 0.3, 'C': 0.1, 'gamma': 0.03}
# FNWN	0.3460	0.1165	0.4511	{'epsilon': 1, 'C': 0.3, 'gamma': 0.01}	0.4425	0.3651	{'epsilon': 1, 'C': 0.01, 'gamma': 0.01}
# OnWN	0.4786	0.1165	0.6070	{'epsilon': 1, 'C': 0.3, 'gamma': 0.01}	0.6147	0.5742	{'epsilon': 1, 'C': 0.01, 'gamma': 0.1}
# Mean: 0.4664	0.1999	0.4946	nan	nan

# Fixed C grid results
# headlines	0.6821	0.1143	0.6946	{'epsilon': 0.1, 'C': 1, 'gamma': 0.01}	0.6946	0.6697	{'epsilon': 0.1, 'C': 1, 'gamma': 0.01}
# SMT	0.3588	0.5149	0.3390	{'epsilon': 0.3, 'C': 1, 'gamma': 0.3}	0.3768	0.4006	{'epsilon': 0.3, 'C': 1, 'gamma': 0.01}
# FNWN	0.3460	0.1143	0.2910	{'epsilon': 0.1, 'C': 1, 'gamma': 0.01}	0.4640	0.3817	{'epsilon': 1, 'C': 1, 'gamma': 0.01}
# OnWN	0.4786	0.1143	0.4776	{'epsilon': 0.1, 'C': 1, 'gamma': 0.01}	0.5959	0.5456	{'epsilon': 1, 'C': 1, 'gamma': 0.01}
# Mean: 0.4664	0.2144	0.4505	0.5328	0.4994


import os

from numpy import vstack, hstack, loadtxt, mean
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR

from ntnu.feats import takelab_feats, takelab_lsa_feats
from ntnu.io import postprocess
from ntnu.sts12 import read_train_data, read_test_data
from ntnu.sts13 import read_blind_test_data
from sts.io import repos_dir, read_system_input
from sts.score import correlation
from sts.sts12 import train_ids, test_ids
from sts.sts13 import test_input_fnames


# number of parallel jobs
N_JOBS = 1

# different size parameter grids

LARGE_PARAM_GRID = {
    'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    'epsilon': [0.01, 0.03, 0.1, 0.3, 1, 3],
    'gamma': [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
}

PARAM_GRID = {
    'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    'epsilon': [0.03, 0.1, 0.3, 1],
    'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
}

SMALL_PARAM_GRID = {
    'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1]
}

FIXED_C_GRID = {
    'C': [1],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1]
}

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

feats = takelab_feats + takelab_lsa_feats

scores = []

for sts12_train_id, sts12_test_id, sts13_test_id in id_pairs:

    param_grid = PARAM_GRID

    # combine 2012 training and test data
    X_sts12_train, y_sts12_train = read_train_data(sts12_train_id, feats)
    X_sts12_test, y_sts12_test = read_test_data(sts12_test_id, feats)
    X_train = vstack([X_sts12_train, X_sts12_test])
    y_train = hstack([y_sts12_train, y_sts12_test])

    X_test = read_blind_test_data(sts13_test_id, feats)

    gs = loadtxt(os.path.join(repos_dir, 'data', 'STS2013-test', "STS.gs.%s.txt" % sts13_test_id))

    test_input = read_system_input(test_input_fnames[sts13_test_id])

    # test set validation/eval split
    X_val = X_test[0:X_test.shape[0]/2, :]
    X_held = X_test[X_test.shape[0]/2:, :]
    gs_val = gs[0:len(gs)/2]
    gs_held = gs[len(gs)/2:]
    test_input_val = test_input[0:len(test_input)/2]
    test_input_held = test_input[len(test_input)/2:]

    # sizes to set up GridSearchCV folds
    n_train = len(y_train)
    n_test = len(gs)

    # picklable scoring function stub with postprocessing step for test set cv
    def score_stub(true, pred):
        return correlation(true, postprocess(test_input_val, pred))

    # Basic training
    regressor = SVR()
    regressor.fit(X_train, y_train)
    y_test = regressor.predict(X_test)
    y_test = postprocess(test_input,  y_test)

    basic_score = correlation(gs, y_test)

    # grid cv on train
    # we can't set up postprocessing internally in the cv here
    grid = GridSearchCV(SVR(), param_grid, cv=5, verbose=1, n_jobs=N_JOBS,
                        scoring=make_scorer(correlation))
    grid.fit(X_train, y_train)
    train_cv_score = grid.best_score_
    train_cv_params = grid.best_params_

    regressor = SVR(**train_cv_params)
    regressor.fit(X_train, y_train)

    y_test = regressor.predict(X_test)
    y_test = postprocess(test_input,  y_test)

    train_cv_held_score = correlation(gs, y_test)

    # grid cv on test held
    cv_split = [(range(n_train), range(n_train, n_train + n_test))]
    grid = GridSearchCV(SVR(), param_grid, cv=cv_split, verbose=1, n_jobs=N_JOBS,
                        scoring=make_scorer(score_stub))
    grid.fit(vstack([X_train, X_test]), hstack([y_train, gs]))
    held_cv_score = grid.best_score_
    held_cv_params = grid.best_params_

    regressor = SVR(**held_cv_params)
    regressor.fit(X_train, y_train)

    y_test = regressor.predict(X_held)
    y_test = postprocess(test_input_held,  y_test)

    held_cv_held_score = correlation(gs_held, y_test)

    # collect scores and print summary
    scores.append((basic_score, train_cv_score, train_cv_held_score, held_cv_score, held_cv_held_score))

    print "%s\t%.04f\t%.04f\t%.04f\t%s\t%.04f\t%.04f\t%s" % \
          (sts13_test_id, basic_score, train_cv_score, train_cv_held_score, train_cv_params,
           held_cv_score, held_cv_held_score, held_cv_params)

print "Mean: %.04f\t%.04f\t%.04f\t%.04f\t%.04f" % tuple(mean(scores, axis=0))
