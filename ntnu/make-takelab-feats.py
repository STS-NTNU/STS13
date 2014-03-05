#!/usr/bin/env python

"""
make Takelab's features for STS training and test data
"""

from os.path import join, exists
from os import makedirs

import sts
import ntnu
import takelab.simpfeats as tl


# requires Takelab LSA models
with_lsa = True

# load word counts for IC weighting
tl.wweight = tl.load_wweight_table("../_data/wordfreq/wordfreq-STS.txt")
tl.minwweight = min(tl.wweight.values())
    
if with_lsa:    
    # load vector spaces    
    tl.nyt_sim = tl.Sim('../_data/lsa-matrices/nyt-words-sts.txt', 
                        '../_data/lsa-matrices/nyt-matrix-sts.txt')
    tl.wiki_sim = tl.Sim('../_data/lsa-matrices/wiki-words-sts.txt', 
                         '../_data/lsa-matrices/wiki-matrix-sts.txt')

    
def make_feats(ids2fnames, dest_dir, with_lsa=True):
    for data_id, input_fname in ids2fnames.items():
        print "creating training features for", data_id
        data_dir= data_id[9:] if data_id.startswith("surprise.") else data_id
        out_dir = join(dest_dir, data_dir)
        if not exists(out_dir):
            makedirs(out_dir)
        
        tl.generate_feats_ntnu(
            input_fname, 
            out_dir,
            with_lsa=with_lsa)

        
#make_feats(sts.sts12.train_input_fnames, 
           #ntnu.sts12.train_dir,
           #with_lsa)

#make_feats(sts.sts12.test_input_fnames, 
           #ntnu.sts12.test_dir,
           #with_lsa)

#make_feats(sts.sts13.test_input_fnames, 
           #ntnu.sts13.test_dir,
           #with_lsa)


make_feats(sts.sts14.trial_input_fnames, 
           ntnu.sts14.trial_dir,
           with_lsa)
