"""
define dirs, ids and filenames for STS13 test data
"""
import os

from os.path import join
from ntnu.io import read_data, map_id_to_feat_files, feat_dir

from sts.io import data_dir, id2filenames

# directory containing original STS 2103 test files
test_dir = join(data_dir, "STS2013-test")

# identifiers for different categories of test data
test_ids = "FNWN", "headlines", "OnWN", "SMT"

# mapping from test identifiers to corresponding input filenames
test_input_fnames = id2filenames(test_dir, "input", test_ids)

# mapping from test identifiers to corresponding gold standard filenames
test_gs_fnames = id2filenames(test_dir, "gs", test_ids)

test_feat_fnames = map_id_to_feat_files(os.path.join(feat_dir, 'STS2013-test'), test_ids)

def read_test_data(ids, features=[], convert_nan=True):
    """
    Create feature vectors and labels for given dataset identifiers and
    features from STS12 test data
    """
    return read_data(ids, test_feat_fnames, test_gs_fnames,
                     features=features, convert_nan=convert_nan )
