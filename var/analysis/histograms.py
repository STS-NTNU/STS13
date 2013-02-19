#!/usr/bin/env python

"""
make histograms of STS12 training and test data
"""

import numpy as np
import matplotlib.pyplot as plt

from sts.io import read




def histograms(data_sets, file_path, fig_fname=None):
    fig = plt.figure()

    for ax_n, data in enumerate(data_sets):
        table = read(file_path.format("input", data), 
                     file_path.format("gs", data))

        ax = fig.add_subplot(2, 3, ax_n+1)
        bins = np.arange(0, 6, 0.5000001)
        ax.hist(table["gold"], bins=bins, facecolor='green', alpha=0.5)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(data)
        ax.set_xlim(0, 5)
        
        ax.grid(True)

    fig.tight_layout()
    
    if fig_fname:
        plt.savefig(fig_fname)
    else:
        plt.show()


histograms(
    data_sets = ("MSRpar", "MSRvid", "SMTeuroparl"),
    file_path = "../../data/STS2012-train/STS.{}.{}.txt",
    fig_fname = "histogram-train.png")
    
histograms(
    data_sets = ("MSRpar", "MSRvid", "SMTeuroparl",
                 "surprise.OnWN", "surprise.SMTnews"
                 ),
    file_path = "../../data/STS2012-test/STS.{}.{}.txt",
    fig_fname = "histogram-test.png")
