"""
common code for reading and writing features
"""

from glob import glob
from os.path import join, normpath, basename, splitext

import numpy as np

from sts.io import read_gold_standard


# find root of Git repos
repos_dir = normpath(join(__file__, "../../../.."))


def feat2filename(dir, id):
    """
    create dict mapping feature names to their filenames
    """
    # a hack, beacuse dir "out/SMTnews" should have been "out/surprise.SMTnews"
    if id.startswith("surprise."):
        id = id[9:]
    pattern = join(dir, id, "*")
    return {splitext(basename(fname))[0]: fname for fname in glob(pattern)}



def read_features(feat_fnames, num_vals, dtype="f"):
    num_feats = len(feat_fnames)
    X = np.empty((num_vals, num_feats), dtype)
    for n, fname in enumerate(feat_fnames):
        X[:,n] = np.fromfile(fname, dtype, num_vals, "\n")
    return X


def read_data(ids, feat_fnames, gs_fnames, features=[], convert_nan=True):
    if isinstance(ids, basestring):
        ids = [ids]

    y = read_gold_standard(gs_fnames[ids[0]])["gold"]
    
    if features:
        # select filenames for desired features
        filenames = [feat_fnames[ids[0]][f] for f in features]
    else:
        # default to using all features for dataset
        filenames = feat_fnames[ids[0]].values()
        
    X = read_features(filenames, num_vals=len(y))
    
    for id in ids[1:]:
        y2 = read_gold_standard(gs_fnames[id])["gold"]
        y = np.hstack([y, y2])
        X2 = read_features(filenames, num_vals=len(y2))
        X = np.vstack([X, X2])
        
    if convert_nan:
        X = np.nan_to_num(X)        
        
    return X, y


def read_blind_data(ids, feat_fnames, features=[], convert_nan=True):
    """
    read data without gold standard
    """
    if isinstance(ids, basestring):
        ids = [ids]
    
    if features:
        # select filenames for desired features
        filenames = [feat_fnames[ids[0]][f] for f in features]
    else:
        # default to using all features for dataset
        filenames = feat_fnames[ids[0]].values()
        
    # HACK: figure out number of values
    num_vals = len(open(filenames[0]).readlines())
        
    X = read_features(filenames, num_vals=num_vals)
    
    for id in ids[1:]:
        X2 = read_features(filenames, num_vals=len(y2))
        X = np.vstack([X, X2])
        
    if convert_nan:
        X = np.nan_to_num(X)        
        
    return X


def postprocess(sys_inp, sys_scores):
    # TODO: this is the Takelab postprocess. The DKPro postprocessing also
    # stripped all characters off the texts which are not in the character
    # range [a-zA-Z0-9]. score is always 5 if sentences are ientical
    sys_scores[sys_inp["s1"] == sys_inp["s2"]] = 5.0 
    # trim scores to range [0,5]
    sys_scores[sys_scores > 5.0] = 5.0 
    sys_scores[sys_scores < 0.0] = 0.0     
    
    
    
    