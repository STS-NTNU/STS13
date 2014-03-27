"""
define dirs, ids and filenames for STS14 trial data
"""
import os

from os.path import join
from ntnu.io import map_id_to_feat_files, read_data, read_blind_data, feat_dir

from sts.io import data_dir, id2filenames


# directory containing original STS 2014 trial files
trial_dir = join(data_dir, "STS2014-trial")

# identifiers for different categories of trial data
trial_ids = "deft-forum", "deft-news", "headlines", "images", "OnWN", "tweet-news"

# mapping from trial identifiers to corresponding input filenames
trial_input_fnames = id2filenames(trial_dir, "input", trial_ids)

# mapping from trial identifiers to corresponding gold standard filenames
trial_gs_fnames = id2filenames(trial_dir, "gs", trial_ids)

# mappings, ids and path for the 2014 test data
test_dir = join(data_dir, "STS2014-test")
test_ids = trial_ids

test_input_fnames = id2filenames(test_dir, "input", test_ids)
test_gs_fnames = id2filenames(test_dir, "gs", test_ids)

# mapping from test dataset identifiers and feature names
# to the corresponding feature files
test_feat_fnames = map_id_to_feat_files(os.path.join(feat_dir, 'STS2014-test'), test_ids)

def read_test_data(ids, features=[], convert_nan=True):
    """
    Create feature vectors and labels for given dataset identifiers and
    features from STS13 test data
    """
    return read_data(ids, test_feat_fnames, test_gs_fnames,
                     features=features, convert_nan=convert_nan)


def read_blind_test_data(ids, features=[], convert_nan=True):
    return read_blind_data(ids, test_feat_fnames, features=features,
                           convert_nan=convert_nan )
