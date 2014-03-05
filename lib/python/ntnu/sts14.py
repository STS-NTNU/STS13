"""
define dirs and filenames of features for STS14 data
"""

from os.path import join

from sts.sts14 import trial_ids, trial_gs_fnames

from ntnu.io import feat_dir, read_data, read_blind_data, map_id_to_feat_files


# top directory containing trial feature files
trial_dir = join(feat_dir, "STS2014-trial")

# mapping from test dataset identifiers and feature names 
# to the corresponding feature files
trial_feat_fnames = map_id_to_feat_files(trial_dir, trial_ids)


def read_trial_data(ids, features=[], convert_nan=True):    
    """
    Create feature vectors and labels for given dataset identifiers and
    features from STS14 trial data
    """    
    return read_data(ids, trial_feat_fnames, trial_gs_fnames,
                     features=features, convert_nan=convert_nan)


def read_blind_trial_data(ids, features=[], convert_nan=True):
    return read_blind_data(ids, trial_feat_fnames, features=features,
                           convert_nan=convert_nan )


    

    