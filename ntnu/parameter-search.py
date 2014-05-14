# Attempts at paramter search for STS14
#
# Trains three models:
# 1) Model with default parameters.
# 2) Model with parameters optimized by CV on th training test.
# 3) Model with parameters optimized by evaluating on a validation part of the test set. The first half of the test
#    set is used for parameter search and the second for evaluation.
#
# Based on the ntnu-template.py code.

# Results aren't really that interesting, bur here they are for the Takelab + subsem features.

# REDO with gamma 0.0

# Large grid results
# headlines	0.7466	0.6876	0.7328	{'epsilon': 0.1, 'C': 0.0001, 'gamma': 0.1}	0.7337	0.7069	{'epsilon': 0.03, 'C': 0.0001, 'gamma': 0.1}
# SMT	0.3769	0.6167	0.2585	{'epsilon': 1, 'C': 0.003, 'gamma': 0.03}	0.3885	0.4070	{'epsilon': 1, 'C': 1000, 'gamma': 0.0003}
# FNWN	0.4105	0.6876	0.3945	{'epsilon': 0.1, 'C': 0.0001, 'gamma': 0.1}	0.4493	0.4066	{'epsilon': 0.3, 'C': 30, 'gamma': 1e-05}
# OnWN	0.6923	0.6876	0.6163	{'epsilon': 0.1, 'C': 0.0001, 'gamma': 0.1}	0.6409	0.6079	{'epsilon': 1, 'C': 0.0001, 'gamma': 3}
# Mean: 0.5566	0.6698	0.5005	0.5531	0.5321

# Small grid results
# headlines	0.7466	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.0}	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.0}
# SMT	0.3769	0.6146	0.2738	{'epsilon': 1, 'C': 0.01, 'gamma': 0.03}	0.3826	0.3961	{'epsilon': 0.3, 'C': 0.01, 'gamma': 0.0}
# FNWN	0.4105	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.0}	0.4464	0.3959	{'epsilon': 0.1, 'C': 0.01, 'gamma': 0.01}
# OnWN	0.6923	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.0}	nan	nan	{'epsilon': 3, 'C': 0.01, 'gamma': 0.0}
# Mean: 0.5566	nan	nan	nan	nan

# Fixed C grid results
# headlines	0.7466	0.7950	0.7071	{'epsilon': 0.1, 'C': 1, 'gamma': 0.3}	0.7567	0.7274	{'epsilon': 1, 'C': 1, 'gamma': 0.01}
# SMT	0.3769	0.7071	0.3256	{'epsilon': 0.3, 'C': 1, 'gamma': 0.3}	0.3787	0.3818	{'epsilon': 0.3, 'C': 1, 'gamma': 0.01}
# FNWN	0.4105	0.7950	0.3705	{'epsilon': 0.1, 'C': 1, 'gamma': 0.3}	0.4382	0.4246	{'epsilon': 1, 'C': 1, 'gamma': 0.01}
# OnWN	0.6923	0.7950	0.6768	{'epsilon': 0.1, 'C': 1, 'gamma': 0.3}	0.6952	0.6791	{'epsilon': 1, 'C': 1, 'gamma': 0.1}
# Mean: 0.5566	0.7730	0.5200	0.5672	0.5532


from numpy import vstack, hstack, concatenate
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR

from ntnu.feats import takelab_feats, takelab_lsa_feats, subsem_best_feats
from ntnu.io import postprocess
from ntnu.sts12 import read_train_data, read_test_data
from sts import sts13
from sts.io import read_system_input
from sts.score import correlation
from sts.sts12 import train_ids, test_ids
from sts.sts13 import test_input_fnames



# number of parallel jobs
N_JOBS = 4

# different size parameter grids

LARGE_PARAM_GRID = {
    'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    'epsilon': [0.01, 0.03, 0.1, 0.3, 1, 3],
    'gamma': [0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
}

PARAM_GRID = {
    'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    'epsilon': [0.03, 0.1, 0.3, 1],
    'gamma': [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
}

SMALL_PARAM_GRID = {
    'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.0, 0.01, 0.03, 0.1, 0.3, 1]
}

FIXED_C_GRID = {
    'C': [1],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.0, 0.01, 0.03, 0.1, 0.3, 1]
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

feats = takelab_feats + takelab_lsa_feats + subsem_best_feats

scores = []

X_sts12_train, y_sts12_train = read_train_data(train_ids, feats)
X_sts12_test, y_sts12_test = read_test_data(test_ids, feats)
X_train = vstack([X_sts12_train, X_sts12_test])
y_train = hstack([y_sts12_train, y_sts12_test])

test_input = [read_system_input(test_input_fnames[sts13_test_id]) for sts13_test_id in sts13.test_ids]
test_input = concatenate(test_input)

X_sts13, y_sts13 = sts13.read_test_data(sts13.test_ids, feats)

X_sts13_val = X_sts13[0:X_sts13.shape[0]/2, :]
X_sts13_held = X_sts13[X_sts13.shape[0]/2:, :]

y_sts_val = y_sts13[0:len(y_sts13)/2]
y_sts_held = y_sts13[len(y_sts13)/2:]

test_input_val = test_input[0:len(test_input)/2]
test_input_held = test_input[len(test_input)/2:]

n_train = len(y_train)
n_test = len(y_sts_val)

param_grid = LARGE_PARAM_GRID

def score_stub(true, pred):
    return correlation(true, postprocess(test_input_val, pred))

# grid cv on test held
cv_split = [(range(n_train), range(n_train, n_train + n_test))]
grid = GridSearchCV(SVR(), param_grid, cv=cv_split, verbose=1, n_jobs=N_JOBS,
                    scoring=make_scorer(correlation))
grid.fit(vstack([X_train, X_sts13_val]), hstack([y_train, y_sts_val]))

held_cv_score = grid.best_score_
held_cv_params = grid.best_params_

print held_cv_score
print held_cv_params

regressor = SVR(**held_cv_params)
regressor.fit(X_train, y_train)

y_test = regressor.predict(X_sts13_held)
# y_test = postprocess(test_input_held,  y_test)

held_cv_held_score = correlation(y_sts_held, y_test)

print held_cv_held_score
