"""
define dirs, ids and filenames for STS14 trial data
"""

from os.path import join

from sts.io import data_dir, id2filenames


# directory containing original STS 2014 trial files
trial_dir = join(data_dir, "STS2014-trial")

# identifiers for different categories of trial data
trial_ids = "deft-forum", "deft-news", "headlines", "images", "OnWN", "tweet-news"

# mapping from trial identifiers to corresponding input filenames
trial_input_fnames = id2filenames(trial_dir, "input", trial_ids)

# mapping from trial identifiers to corresponding gold standard filenames
trial_gs_fnames = id2filenames(trial_dir, "gs", trial_ids)
