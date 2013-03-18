"""
define dirs and filenames of features for STS12 data
"""
from os.path import join

from sts.sts12 import train_ids, test_ids, train_gs_fnames, test_gs_fnames

from ntnu.io import repos_dir, read_data, feat2filename


train_dir = join(repos_dir, "out/STS2012-train")
test_dir = join(repos_dir, "out/STS2012-test")

train_feat_fnames = {id: feat2filename(train_dir, id) for id in train_ids}
test_feat_fnames = {id: feat2filename(test_dir, id) for id in test_ids}

# Example usage:
# >>> train_feat_fnames["MSRpar"]["LongestCommonSubsequenceComparator"]
# '/Users/erwin/Projects/SemTextSim/github/STS13/out/STS2012-train/MSRpar/LongestCommonSubsequenceComparator.txt'


def read_train_data(ids, features=[], convert_nan=True):
    return read_data(ids, train_feat_fnames, train_gs_fnames,
                     features=features, convert_nan=convert_nan)


def read_test_data(ids, features=[], convert_nan=True):
    return read_data(ids, test_feat_fnames, test_gs_fnames, 
                     features=features, convert_nan=convert_nan )

    

    