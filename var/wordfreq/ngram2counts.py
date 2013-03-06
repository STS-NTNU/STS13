#!/usr/bin/env python

"""
convert Google 1-gram counts broken down per year to simple word counts
(removing year and document count)

Usage:

./ngram2counts.py googlebooks-eng-all-1gram-20120701-*.gz 
"""

import gzip
import sys


def get_counts(infname, outfname):
    outf = gzip.open(outfname, "w")
    
    for l in gzip.open(infname):
        word, _, count , _ = l.split("\t")
        try:
            if word == prev_word:
                total += int(count)
            else:
                outf.write("{}\t{}\n".format(prev_word, total))
                prev_word = word
                total = 0
        except NameError:
            prev_word = word
            total = int(count)
    
    outf.write("{}\t{}n".format(prev_word, total))
    outf.close()
        
    
    
for infname in sys.argv[1:]:
    print "processing", infname
    outfname = "freq-" + infname
    get_counts(infname, outfname)
    print "created", outfname
    
