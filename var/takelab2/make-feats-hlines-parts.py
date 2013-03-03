#!/usr/bin/env python

"""
make features for part of hlines
"""

import sys

import takelab.simpfeats as tl


# load word counts for IC weighting
tl.wweight = tl.load_wweight_table("../wordfreq/_wordfreq_hlines_parts.txt")
tl.minwweight = min(tl.wweight.values())

for part in "abcdefghij":
    out_fname = "_npz_data/_hlines_part_a{}.npz".format(part)
    sys.stderr.write("creating {}\n".format(out_fname))
    tl.generate_features("../../../../local/corpora/hlines/_hlines_part_a{}".format(part),
                         outf=out_fname, 
                         out_format="numpy", 
                         with_lsa=False)
    