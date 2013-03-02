#!/usr/bin/env python

"""
make features for STS12 training and test data
"""

import sys

import takelab.simpfeats as tl


# load word counts for IC weighting
tl.wweight = tl.load_wweight_table("../wordfreq/_wordfreq-STS2012.txt")
tl.minwweight = min(tl.wweight.values())

with_lsa = False    

# load vector spaces    
if with_lsa:
    tl.nyt_sim = tl.Sim('_vsm_data/nyt_words.txt', '_vsm_data/nyt_word_vectors.txt')
    tl.wiki_sim = tl.Sim('_vsm_data/wikipedia_words.txt', '_vsm_data/wikipedia_word_vectors.txt')

# create training instances
train_dir = "../../data/STS2012-train"

for data in "MSRpar", "MSRvid", "SMTeuroparl":
    out_fname = "_npz_data/_STS2012.train.{}.npz".format(data)
    sys.stderr.write("creating {}\n".format(out_fname))
    tl.generate_features("{}/STS.input.{}.txt".format(train_dir, data),
                         "{}/STS.gs.{}.txt".format(train_dir, data),
                         outf=out_fname, 
                         out_format="numpy", 
                         with_lsa=with_lsa)
    
# create test instances
test_dir =  "../../data/STS2012-test"  

for data in "MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews":
    out_fname = "_npz_data/_STS2012.test.{}.npz".format(data)
    sys.stderr.write("creating {}\n".format(out_fname))
    tl.generate_features("{}/STS.input.{}.txt".format(test_dir, data),
                         "{}/STS.gs.{}.txt".format(test_dir, data),
                         outf=out_fname, 
                         out_format="numpy", 
                         with_lsa=with_lsa)