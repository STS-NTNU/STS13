#!/usr/bin/env python

"""
make features for STS training and test data for use with NTNU system
"""

import sys
from os.path import join, exists
from os import makedirs

import takelab.simpfeats as tl

from sts import sts12, sts13

# load word counts for IC weighting
tl.wweight = tl.load_wweight_table("../_data/wordfreq/wordfreq-STS.txt")
tl.minwweight = min(tl.wweight.values())
   

# load vector spaces    
tl.nyt_sim = tl.Sim('../_data/lsa-matrices/nyt-words-sts.txt', 
                    '../_data/lsa-matrices/nyt-matrix-sts.txt')
tl.wiki_sim = tl.Sim('../_data/lsa-matrices/wiki-words-sts.txt', 
                     '../_data/lsa-matrices/wiki-matrix-sts.txt')

with_lsa = True


dest_dir = "../out/STS2012-train"

for data_id in sts12.train_ids:
    print "creating training features for", data_id
    data_dir= data_id[9:] if data_id.startswith("surprise.") else data_id
    out_dir = join(dest_dir, data_dir)
    if not exists(out_dir):
        makedirs(out_dir)
    
    tl.generate_feats_ntnu(
        sts12.train_input_fnames[data_id], 
        out_dir,
        with_lsa=with_lsa)

dest_dir = "../out/STS2012-test"

for data_id in sts12.test_ids:
    print "creating test features for", data_id
    data_dir= data_id[9:] if data_id.startswith("surprise.") else data_id
    out_dir = join(dest_dir, data_dir)
    if not exists(out_dir):
        makedirs(out_dir)
    
    tl.generate_feats_ntnu(
        sts12.test_input_fnames[data_id], 
        out_dir,
        with_lsa=with_lsa)
    
dest_dir = "../out/STS2013-test"

for data_id in sts13.test_ids:
    print "creating test features for", data_id
    data_dir= data_id[9:] if data_id.startswith("surprise.") else data_id
    out_dir = join(dest_dir, data_dir)
    if not exists(out_dir):
        makedirs(out_dir)
    
    tl.generate_feats_ntnu(
        sts13.test_input_fnames[data_id], 
        out_dir,
        with_lsa=with_lsa)
