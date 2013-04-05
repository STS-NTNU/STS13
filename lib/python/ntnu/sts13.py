"""
define dirs and filenames of features for STS13 data
"""
from os.path import join

from sts.sts13 import test_ids, test_gs_fnames

from ntnu.io import repos_dir, read_data, read_blind_data, feat2filename


#train_dir = join(repos_dir, "out/STS2013-train")
test_dir = join(repos_dir, "out/STS2013-test")

#train_feat_fnames = {id: feat2filename(train_dir, id) for id in train_ids}
test_feat_fnames = {id: feat2filename(test_dir, id) for id in test_ids}


#def read_train_data(ids, features=[], convert_nan=True):
#    return read_data(ids, train_feat_fnames, train_gs_fnames,
#                     features=features, convert_nan=convert_nan)


def read_blind_test_data(ids, features=[], convert_nan=True):
    return read_blind_data(ids, test_feat_fnames, features=features,
                           convert_nan=convert_nan )

def read_test_data(ids, features=[], convert_nan=True):
    return read_data(ids, test_feat_fnames, test_gs_fnames, 
                     features=features, convert_nan=convert_nan)


    

    