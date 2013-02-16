#!/usr/bin/env python

"""

"""

import numpy as np

from sts.io import read
from sts.score import correlation

import matplotlib.pyplot as plt


for data in "MSRpar", "MSRvid", "SMTeuroparl":
    table = read("../../data/STS2012-test/STS.input.{}.txt".format(data),
                 "../../data/STS2012-test/STS.gs.{}.txt".format(data),
                 "takelab-out/{}-output.txt".format(data.lower()))
    
    x = table["gold"]
    y = table["output"]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    y2 = m*y + c
    
    print m, c    
    print correlation(x,y) - correlation(x,y2)
    
    plt.plot(x, y, '.')
    plt.plot(x, y2, 'xr')
    #plt.legend()
    plt.xlim(0,5)
    plt.ylim(0,5)
    plt.show()