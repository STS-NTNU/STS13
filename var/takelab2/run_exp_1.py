from sklearn.svm import SVR

import numpy as np

from sts.io import read_system_input, read_gold_standard
from sts.score import correlation


train = np.load("_npz_data/_STS2012.train.MSRpar.npz")
clf = SVR(kernel='rbf', C=50, epsilon=.2, gamma=.02)
print clf

clf.fit(train["X"], train["y"])
#print clf.score(train["X"], train["y"])
    
test = np.load("_npz_data/_STS2012.test.MSRpar.npz")
#print clf.score(test["X"], test["y"])
sys_scores = clf.predict(test["X"])

# postprocess
sys_inp = read_system_input("../../data/STS2012-test/STS.input.MSRpar.txt")
sys_scores[sys_inp["s1"] == sys_inp["s2"]] = 5.0 
sys_scores[sys_scores > 5.0] = 5.0 
sys_scores[sys_scores < 0.0] = 0.0 

# compute correlation score
gold_scores = read_gold_standard("../../data/STS2012-test/STS.gs.MSRpar.txt")["gold"]
print correlation(gold_scores, sys_scores)


#from sklearn.cross_validation import KFold
#from sklearn.grid_search import GridSearchCV
    
#C_range = 10.0 ** np.arange(-2, 9)
#gamma_range = 10.0 ** np.arange(-5, 4)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = KFold(train["y"].size, k=3, shuffle=True)
#grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=cv)
#grid.fit(train["X"], train["y"])

#print("The best classifier is: ", grid.best_estimator_)
