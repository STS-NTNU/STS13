"""
common code for reading and writing features
"""

from glob import glob
from os.path import join, basename, splitext

import numpy as np

from sts.io import read_gold_standard, repos_dir


# main directory for feature files
feat_dir = join(repos_dir, "out")



def map_id_to_feat_files(dir, ids):
    """
    Create a mapping from dataset identifiers and feature names to the
    corresponding feature files.
    
    Parameters
    ----------
    dir: str
        main directory containing subdirs for dataset identifiers
    ids: list of str
        identifiers/subdirs for datasets
        
    Returns
    -------
    dict:
         mapping from dataset identifiers to mapping from feature names to 
         their files
         
         
    Example
    -------
    >>> ntnu.io.map_id_to_feat_files("out/STS2012-test", ["MSRpar", "MSRvid"])
    {'MSRpar': {'CharacterNGramMeasure_2':
                'out/STS2012-test/MSRpar/CharacterNGramMeasure_2.txt',
                 'CharacterNGramMeasure_3':
                 'out/STS2012-test/MSRpar/CharacterNGramMeasure_3.txt',
                 ...
            }
     'MSRvid': {'CharacterNGramMeasure_2':
                'out/STS2012-test/MSRvid/CharacterNGramMeasure_2.txt',
                'CharacterNGramMeasure_3': 
                'out/STS2012-test/MSRvid/CharacterNGramMeasure_3.txt',
                ...
            }            
    }
     
    """
    return {id: map_feat_to_files(dir, id) 
            for id in ids}
    
    

def map_feat_to_files(dir, id):
    """
    Create a mapping from feature names to their files.
    
    Parameters
    ----------
    dir: str
        main directory containing feature files
    id: str
        identifier/subdir for dataset
        
    Returns
    -------
    dict:
         mapping from feature names to their files
         
    Example
    -------
    >>> feat2filename("out/STS2012-test","MSRpar")
    {'tl.n_gram_match_lem_3': 
     'out/STS2012-test/MSRpar/tl.n_gram_match_lem_3.txt',  
     'tl.n_gram_match_lem_2': 
     'out/STS2012-test/MSRpar/tl.n_gram_match_lem_2.txt', 
     'tl.n_gram_match_lem_1': 
     'out/STS2012-test/MSRpar/tl.n_gram_match_lem_1.txt', 
     'RI_HungarianAlgorithm_Measure': 
     'out/STS2012-test/MSRpar/RI_HungarianAlgorithm_Measure.txt', 
     'tl.stocks_match_len': 
     'out/STS2012-test/MSRpar/tl.stocks_match_len.txt', 
     ...
    }
    """
    # a hack, because dir "out/SMTnews" should have been "out/surprise.SMTnews"
    if id.startswith("surprise."):
        id = id[9:]
    pattern = join(dir, id, "*")
    return {splitext(basename(fname))[0]: fname 
            for fname in glob(pattern)}


def read_data(ids, feat_fnames, gs_fnames, features=[], convert_nan=True):
    """
    Create feature vectors and labels. 
    
    Parameters
    ----------
    ids: str or list of str
        dataset identifier(s)
    feat_names: list of str
        mapping from feature nams to feature files
    gs_fnames: list of str
        gold standard filenames
    features:
        feature names
    convert_nan: True or False
        replace nan with zero and inf with finite numbers in feature values
    
    Returns
    -------
    X, y: numpy.array, numpy.array
        2-dimesional array of feature values and 1-dimensional array of labels,
        intended for use with sklearn     
    """
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
        if features:
            # select filenames for desired features
            filenames = [feat_fnames[id][f] for f in features]
        else:
            # default to using all features for dataset
            filenames = feat_fnames[id].values()

        y2 = read_gold_standard(gs_fnames[id])["gold"]
        y = np.hstack([y, y2])
        X2 = read_features(filenames, num_vals=len(y2))
        X = np.vstack([X, X2])
        
    if convert_nan:
        X = np.nan_to_num(X)        
        
    return X, y


def read_features(feat_fnames, num_vals, dtype="f"):
    num_feats = len(feat_fnames)
    X = np.empty((num_vals, num_feats), dtype)
    for n, fname in enumerate(feat_fnames):
        X[:,n] = np.fromfile(fname, dtype, num_vals, "\n")
    return X


# FIXME: replace with read_data(..., blind=True)
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

    return sys_scores
    
    
    
    