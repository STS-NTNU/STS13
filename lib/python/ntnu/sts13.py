"""
define dirs and filenames of features for STS13 data
"""

from os.path import join

from sts.sts13 import test_ids, test_gs_fnames

from ntnu.io import feat_dir, read_data, read_blind_data, map_id_to_feat_files


# top directory containing test feature files
test_dir = join(feat_dir, "STS2013-test")

# mapping from test dataset identifiers and feature names 
# to the corresponding feature files
test_feat_fnames = map_id_to_feat_files(test_dir, test_ids)


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


    

    