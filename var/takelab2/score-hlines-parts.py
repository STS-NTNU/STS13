import glob

from sklearn.svm import SVR

import numpy as np


from sts.io import read_system_input, read_gold_standard
from sts.score import correlation


train = np.load("_npz_data/_STS2012.train.MSRpar.npz")
clf = SVR(kernel='rbf', C=50, epsilon=.2, gamma=.02)
print clf

clf.fit(train["X"], train["y"])
#print clf.score(train["X"], train["y"])
    
for fname in glob.glob("_npz_data/_hlines_part_???.npz"):   
    print "scoring", fname
    # numpy.load does not release file handle for npz files
    # See http://projects.scipy.org/numpy/ticket/1517
    fd = open(fname, "rab")
    test = np.load(fd)
    y = clf.predict(test["X"])
    np.savez(fname, X=test["X"], y=y)
    fd.close()
    
    
