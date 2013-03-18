#!/usr/bin/env python

"""
create smaller versions of Takelab's LSA models, consisting of words and
matrix files, for only the STS words, because the original versions are too
large to fit into memory
"""

import codecs

from sts import sts12
from sts import sts13
from takelab.simpfeats import load_data, get_lemmatized_words

import nltk


input_fnames = ( sts12.train_input_fnames.values() +
                 sts12.test_input_fnames.values() +
                 sts13.test_input_fnames.values() )

words = set()

for infname in input_fnames:
    print "reading", infname
    for sa, sb in load_data(infname):
        words.update(get_lemmatized_words(sa))
        words.update(get_lemmatized_words(sb))
        

for data in "nyt", "wiki":        
    print "creating {} words and matrix file".format(data)
    
    words_inf = codecs.open("../../_data/lsa-matrices/{}-words.txt".format(data), encoding="utf-8")
    matrix_inf = open("../../_data/lsa-matrices/{}-matrix.txt".format(data))
    words_outf = codecs.open("../../_data/lsa-matrices/{}-words-sts.txt".format(data), "w", encoding="utf-8")
    matrix_outf = open("../../_data/lsa-matrices/{}-matrix-sts.txt".format(data), "w")
    
    for word, vector in zip(words_inf, matrix_inf):
        if word.strip() in words:
            words_outf.write(word)
            matrix_outf.write(vector)
            
    words_outf.close()
    matrix_outf.close()

print "done"


        
        



        
                
         