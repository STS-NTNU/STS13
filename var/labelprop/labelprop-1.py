"""
Trying out label propagation

No luck. I tried with different settings (e.g. alpha), but the score with
using all labeled data are too low to be competetive with e.g. SVR.

I also doubt that this implementation really works with regression, even
though the docs say so.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.semi_supervised import label_propagation

from sts.io import read_system_input
from sts.score import correlation
from sts.sts12 import test_input_fnames, train_ids
from ntnu.sts12 import read_train_data, read_test_data, postprocess, takelab_feats, all_feats


rng = np.random.RandomState(0)

# ids of training and testing data sets
id_pairs = [ 
    ("MSRpar", "MSRpar"),
    ("MSRvid", "MSRvid"),
    ("SMTeuroparl", "SMTeuroparl"),
    ("SMTeuroparl", "surprise.SMTnews"),
    #(train_ids, "surprise.OnWN") 
]

feats = all_feats


for train_id, test_id in id_pairs:
    X_train, y_train = read_train_data(train_id, feats)
    X_test, y_test = read_test_data(test_id, feats)
    sys_input = read_system_input(test_input_fnames[test_id])
    
    # p = percentage unlabeled
    for p in np.arange(0, 1.0, 0.05):
        y = np.copy(y_train)
        y[rng.rand(len(y)) < p] = -1
        clf = label_propagation.LabelSpreading(alpha=0.5)
        #clf = label_propagation.LabelPropagation()
        clf.fit(X_train, y)
        sys_scores = clf.predict(X_test)
        postprocess(sys_input, sys_scores)
        
        print "{:32s} {:32s} {:8.2f} {:>12d} {:8.1f}".format(
            train_id, 
            test_id, 
            p,
            sum(y < 0),
            correlation(sys_scores, y_test) * 100)
        
    print