#!/usr/bin/env python


"""
multiple iterations (does not help)
"""

import glob

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from sts.io import read_system_input, read_gold_standard
from sts.score import correlation
from sts.sts12 import test_input_fnames
from ntnu.sts12 import ( train_ids, read_train_data, read_test_data,
                         postprocess, takelab_feats, takelab_lsa_feats )



# features to be used
feats = takelab_feats
# feats = takelab_feats + takelab_lsa_feats 

# learning algorithms, one per test set, where SVR settings result from
# grid-search.sh
#regressors = {
    #"MSRpar":      SVR( epsilon=0.2, gamma=0.02, C=50),
    #"MSRvid":      SVR( epsilon=0.5, gamma=0.02, C=200),
    #"SMTeuroparl": SVR( epsilon=0.2, gamma=0.02, C=100)
    ##"SMTeuroparl": SVR(C=100, epsilon=0.2, gamma=0.02),
    ##train_ids:     SVR(C=10,  epsilon=0.5, gamma=0.02)
    #}

regressors = {
    "MSRpar":      LinearRegression(),
    "MSRvid":      LinearRegression(),
    "SMTeuroparl": LinearRegression(),
    }


# mapping of test data to their training data
test_id2train_id = { 
    "MSRpar":           "MSRpar",
    "MSRvid":           "MSRvid",
    "SMTeuroparl":      "SMTeuroparl",
    "surprise.SMTnews": "SMTeuroparl",
    "surprise.OnWN":    "MSRpar" 
}

# read all training and test data
X_train = {}
y_train = {}
X_test = {}
y_test = {}
sys_input = {}

for train_id in regressors.keys():
    X_train[train_id], y_train[train_id] = read_train_data(train_id, feats)
    
for test_id in test_id2train_id.keys():
    X_test[test_id], y_test[test_id] = read_test_data(test_id, feats)
    sys_input[test_id] = read_system_input(test_input_fnames[test_id])

# read headlines
n = 25
print "number headline parts:", n
hline_fnames = glob.glob("_npz_data/_hlines_part_???.npz")[:n] 

collect = []
for fn in hline_fnames:
    fh = open(fn)
    collect.append(np.load(fh)["X"])
    fh.close()
    
X_hline = np.vstack(collect)
#X_hline = np.vstack([np.load(fn)["X"] for fn in hline_fnames])

# y_hline stores scores for each regressors 
hline_regressors = "MSRpar", "MSRvid", "SMTeuroparl"
y_dtype = [(id,"f") for id in hline_regressors] + [("mean2", "f")]
y_hline = np.recarray(X_hline.shape[0], dtype=y_dtype)
    
    
results = np.recarray(0, dtype=[("train_id", "S16"), 
                                ("test_id", "S16"),
                                ("min_diff", "f"), 
                                ("max_diff", "f"),
                                ("samples", "i"),
                                ("iteration", "i"),
                                ("score", "f")])    

         
# fit regressor on training data
for train_id, rgr in regressors.items():
    rgr.fit(X_train[train_id], y_train[train_id])
    
# compute and report initial score on test data
for test_id, train_id in test_id2train_id.items():
    rgr = regressors[train_id]
    sys_scores = rgr.predict(X_test[test_id])
    postprocess(sys_input[test_id],  sys_scores)
    r = correlation(sys_scores, y_test[test_id])
    n = X_train[train_id].shape[0]
    results.resize(results.size + 1)
    if isinstance(train_id, tuple):
        train_id = "+".join(train_id)    
    results[-1] = (train_id, test_id, 0, 0, n, 0, r)
    print "{:32s} {:32s} {:>8d} {:8.4f}".format(train_id, test_id, n, r)
   
# score headlines
for train_id in hline_regressors:  
    # TODO: full postprocessing
    scores = regressors[train_id].predict(X_hline)
    scores[scores < 0] = 0.0 
    scores[scores > 5] = 5.0 
    y_hline[train_id] = scores

        
for max_diff in [1.5, 2.0]:  
    for min_diff in [0.25]:
        
        # indices of extra headline samples for each regressor         
        extra = {id: set() for id in regressors}
        
        for iteration in [1,2,3,4,5]:
            print "min_diff:", min_diff 
            print "max_diff:", max_diff
            print "iteration:", iteration
            
            # As long as min_diff < max_diff, the three branches of the if-then statement
            # are exclusive (i.e. we can use "elif"). 
            
            # Also, any sample that is scored "wrong" by a single regressor but "correct"
            # by the other two regressors, can be deleted from the unlabeled samples,
            # because it is added to the training data of the first regressor and it
            # won't help improving the other two regressors.        
            
            for i, scores in enumerate(y_hline):
                if ( abs(scores["MSRpar"] - scores["MSRvid"]) > max_diff and
                     abs(scores["MSRpar"] - scores["SMTeuroparl"]) > max_diff and
                     abs(scores["MSRvid"] - scores["SMTeuroparl"]) < min_diff ):
                    #print "+MSRpar:", s_par, s_vid, s_euro 
                    y_hline[i]["mean2"] = (scores["MSRvid"] + scores["SMTeuroparl"])/2.0 
                    extra["MSRpar"].add(i)
                elif ( abs(scores["MSRvid"] - scores["MSRpar"]) > max_diff and
                       abs(scores["MSRvid"] - scores["SMTeuroparl"]) > max_diff and
                       abs(scores["MSRpar"] - scores["SMTeuroparl"]) < min_diff ):
                    #print "+MSRvid:", s_par, s_vid, s_euro 
                    extra["MSRvid"].add(i)
                    y_hline[i]["mean2"] = (scores["MSRpar"] + scores["SMTeuroparl"])/2.0 
                elif ( abs(scores["SMTeuroparl"] - scores["MSRpar"]) > max_diff and
                       abs(scores["SMTeuroparl"] - scores["MSRvid"]) > max_diff and
                       abs(scores["MSRpar"] - scores["MSRvid"]) < min_diff ):
                    #print "+SMTeuro:", s_par, s_vid, s_euro 
                    extra["SMTeuroparl"].add(i)
                    y_hline[i]["mean2"] = (scores["MSRpar"] + scores["MSRvid"])/2.0         
                    
                
            for test_id, train_id in test_id2train_id.items():
                # fit regressor on training data plus extra headline samples
                X = np.vstack([X_train[train_id],
                               X_hline[list(extra[train_id])]])
                y = np.hstack([y_train[train_id],
                               y_hline["mean2"][list(extra[train_id])]])         
                rgr = regressors[train_id]
                rgr.fit(X, y)
                
                # compute and report score on test data   
                sys_scores = rgr.predict(X_test[test_id])
                postprocess(sys_input[test_id],  sys_scores)
                r = correlation(sys_scores, y_test[test_id])
                n = X.shape[0]
                results.resize(results.size + 1)
                if isinstance(train_id, tuple):
                    train_id = "+".join(train_id)    
                results[-1] = (train_id, test_id, min_diff, max_diff, n, iteration, r)
                print "{:32s} {:32s} {:>8d} {:8.4f}".format(train_id, test_id, n, r)    
                
                
            # re-score headlines
            for train_id, rgr in regressors.items():  
                # TODO: full postprocessing
                scores = rgr.predict(X_hline)
                scores[scores < 0] = 0.0 
                scores[scores > 5] = 5.0 
                y_hline[train_id] = scores
            
    

np.save("_results_bootstrap_hlines_2", results) 