"""
define dirs and filenames of features for STS12 data
"""

from glob import glob
from os.path import join, normpath, basename, splitext

import numpy as np

from sts.io import read_gold_standard
from sts.sts12 import train_ids, test_ids, train_gs_fnames, test_gs_fnames


def feat2filename(dir, id):
    """
    create dict mapping feature names to their filenames
    """
    # a hack, beacuse dir "out/SMTnews" should have ben "out/surprise.SMTnews"
    if id.startswith("surprise."):
        id = id[9:]
    pattern = join(dir, id, "*")
    return {splitext(basename(fname))[0]: fname for fname in glob(pattern)}

# find root of Git repos
repos_dir = normpath(join(__file__, "../../../.."))

train_dir = join(repos_dir, "out/STS2012-train")
test_dir = join(repos_dir, "out/STS2012-test")

train_feat_fnames = {id: feat2filename(train_dir, id) for id in train_ids}
test_feat_fnames = {id: feat2filename(test_dir, id) for id in test_ids}

# Example usage:
# >>> train_feat_fnames["MSRpar"]["LongestCommonSubsequenceComparator"]
# '/Users/erwin/Projects/SemTextSim/github/STS13/out/STS2012-train/MSRpar/LongestCommonSubsequenceComparator.txt'


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


def read_train_data(ids, features=[], convert_nan=True):
    return read_data(ids, train_feat_fnames, train_gs_fnames,
                     features=features, convert_nan=convert_nan)


def read_test_data(ids, features=[], convert_nan=True):
    return read_data(ids, test_feat_fnames, test_gs_fnames, 
                     features=features, convert_nan=convert_nan )

    
def postprocess(sys_inp, sys_scores):
    # TODO: this is the Takelab postprocess. The DKPro postprocessing also
    # stripped all characters off the texts which are not in the character
    # range [a-zA-Z0-9]. score is always 5 if sentences are ientical
    sys_scores[sys_inp["s1"] == sys_inp["s2"]] = 5.0 
    # trim scores to range [0,5]
    sys_scores[sys_scores > 5.0] = 5.0 
    sys_scores[sys_scores < 0.0] = 0.0     
    


    
dkpro_feats = [
    "WordNGramJaccardMeasure_4_stopword-filtered",
    "WordNGramJaccardMeasure_4", 
    "LongestCommonSubsequenceComparator",
    "CharacterNGramMeasure_4", 
    "CharacterNGramMeasure_2", 
    "CharacterNGramMeasure_3",
    "LongestCommonSubsequenceNormComparator", 
    "WordNGramJaccardMeasure_1",
    "WordNGramJaccardMeasure_2_stopword-filtered", 
    "WordNGramJaccardMeasure_3",
    "WordNGramContainmentMeasure_1_stopword-filtered",
    "LongestCommonSubstringComparator", 
    "GreedyStringTiling_3",
    "WordNGramContainmentMeasure_2_stopword-filtered"
]
    

takelab_feats= [ 
    "tl.number_len", "tl.number_f", "tl.number_subset",
    "tl.case_match_len", "tl.case_match_f",
    "tl.stocks_match_len", "tl.stocks_match_f",
    "tl.n_gram_match_lc_1",
    "tl.n_gram_match_lc_2",
    "tl.n_gram_match_lc_3",
    "tl.n_gram_match_lem_1",
    "tl.n_gram_match_lem_2",
    "tl.n_gram_match_lem_3",
    "tl.wn_sim_lem",
    "tl.weight_word_match_olc",
    "tl.weight_word_match_lem",
    "tl.rel_len_diff_lc",
    "tl.rel_ic_diff_olc"
]

takelab_lsa_feats = [
    "tl.dist_sim_nyt",
    "tl.weight_dist_sim_nyt",
    "tl.weight_dist_sim_wiki"
]    

gleb_feats = [
    "SemanticWordOrderLeacockAndChodorow"]


all_feats = (
    dkpro_feats +
    takelab_feats +
    #takelab_lsa_feats +
    gleb_feats
    )