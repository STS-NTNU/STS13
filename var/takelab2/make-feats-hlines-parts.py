#!/usr/bin/python

"""
make features for part of hlines
"""

# NB I ran a variant of this on the Translate server

import sys

sys.path.append("./lib/python")

import takelab.simpfeats as tl


# load word counts for IC weighting
tl.wweight = tl.load_wweight_table("../wordfreq/_wordfreq_hlines.txt")
tl.minwweight = min(tl.wweight.values())

for in_fname in sys.argv[1:]:
    out_fname = in_fname + ".npz"
    sys.stderr.write("creating {}\n".format(out_fname))
    tl.generate_features(in_fname,
                         outf=out_fname, 
                         out_format="numpy", 
                         with_lsa=False)
    
