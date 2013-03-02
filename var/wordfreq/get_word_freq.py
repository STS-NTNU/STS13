#!/usr/bin/env python

"""
get counts for words in train/test material only
"""

import codecs
import glob
import gzip
import sys

import nltk


# file glob for word counts derived from Gogle 1-grams with ngram2count.py
fname_pat = "../../../../local/google-n-grams/freq*.gz"

words = set()

for infname in sys.argv[1:]:
    for l in codecs.open(infname):
        for s in l.split("\t"):
            for w in nltk.word_tokenize(s):
                words.add(w)
                
                
for count_fname in glob.glob(fname_pat):
    for l in gzip.open(count_fname):
        word, count = l.split("\t")
        word = word.decode("utf-8")
        if word in words:
            print word, count,
            
    
    
                
        
        
    




