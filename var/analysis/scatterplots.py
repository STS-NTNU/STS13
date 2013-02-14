#!/usr/bin/env python

"""
make scatterplots of system output on STS12 test data
"""

import numpy as np
import matplotlib.pyplot as plt

from sts.io import read_gold_standard, read_system_output
from sts.score import correlation

# Takelab system

for data in "MSRpar", "MSRvid", "SMTeuroparl":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    gold = read_gold_standard("../../data/STS2012-test/STS.gs.{}.txt".format(data))
    out = read_system_output("takelab-out/{}-output.txt".format(data.lower()))
    
    ax.plot(gold, out, ".")
    r = correlation(gold["gold"], out["output"])
    
    ax.set_xlim(-0.5,5.5)
    ax.set_ylim(-0.5,5.5)
    ax.set_xlabel("Gold")
    ax.set_ylabel("System")
    ax.set_title("TakeLab.TST12.Test.{} (r={})".format(data, r))
    ax.grid(True)
    

    plt.savefig("scatter-takelab-tst12-test-{}.png".format(data))
    plt.show()
