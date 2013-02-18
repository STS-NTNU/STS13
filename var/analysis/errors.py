#!/usr/bin/env python

"""
Extract sentence pairs for which the predicted similarity scores by the
TakeLab simple system deviate the most from the gold standard
"""

import codecs
import numpy as np

from sts.io import read
from sts.score import correlation

import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields


for data in "MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews":
    table = read("../../data/STS2012-test/STS.input.{}.txt".format(data),
                 "../../data/STS2012-test/STS.gs.{}.txt".format(data),
                 "takelab-out/{}-output.txt".format(data.lower()))
    
    # add new filed "diff" for difference between gold standard and system output    
    diff = abs(table["gold"] - table["output"])
    table = append_fields(table, 'diff', diff, 'f', usemask=False)
    # sort descending on diff
    table.sort(axis=0, order='diff')
    table = table[::-1]
    # write in "markdown" format for Github wiki  
    f = codecs.open("Errors.TakeLab." + data + ".txt", "w", encoding="utf-8")
    for i, row in enumerate(table):
        f.write("**{}: ".format(i+1))
        f.write(u"diff={0[4]:.4}, gold={0[2]:.4},  sys={0[3]:.4}**\n\n"
                u"  * {0[0]}\n  * {0[1]}\n\n---\n\n".format(row))
        