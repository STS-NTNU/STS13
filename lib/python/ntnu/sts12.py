"""
define dirs and filenames of features for STS12 data
"""

from os.path import join

from sts.sts12 import train_ids, test_ids, train_gs_fnames, test_gs_fnames

from ntnu.io import feat_dir, read_data, map_id_to_feat_files



# top directory containing train feature files
train_dir = join(feat_dir, "STS2012-train")

# mapping from train dataset identifiers and feature names 
# to the corresponding feature files
train_feat_fnames = map_id_to_feat_files(train_dir, train_ids)

def read_train_data(ids, features=[], convert_nan=True):    
    """
    Create feature vectors and labels for given dataset identifiers and
    features from STS12 train data
    """    
    return read_data(ids, train_feat_fnames, train_gs_fnames,
                     features=features, convert_nan=convert_nan)



# top directory containing test feature files
test_dir = join(feat_dir, "STS2012-test")

# mapping from test dataset identifiers and feature names 
# to the corresponding feature files
test_feat_fnames = map_id_to_feat_files(test_dir, test_ids)

def read_test_data(ids, features=[], convert_nan=True):
    """
    Create feature vectors and labels for given dataset identifiers and
    features from STS12 test data
    """
    return read_data(ids, test_feat_fnames, test_gs_fnames, 
                     features=features, convert_nan=convert_nan )

    

    