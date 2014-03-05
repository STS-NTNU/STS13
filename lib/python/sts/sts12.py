"""
define dirs, ids and filenames for STS12 training and testing data
"""

from os.path import join

from sts.io import data_dir, id2filenames


# directory containing original STS 2102 train files
train_dir = join(data_dir, "STS2012-train")

# identifiers for different categories of train data
train_ids = "MSRpar", "MSRvid", "SMTeuroparl"

# mapping from train identifiers to corresponding input filenames
train_input_fnames = id2filenames(train_dir, "input", train_ids)

# mapping from train identifiers to corresponding gold standard filenames
train_gs_fnames = id2filenames(train_dir, "gs", train_ids)



# directory containing original STS 2102 test files
test_dir = join(data_dir, "STS2012-test")

# identifiers for different categories of test data
test_ids = "MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"

# mapping from test identifiers to corresponding input filenames
test_input_fnames = id2filenames(test_dir, "input", test_ids)

# mapping from test identifiers to corresponding gold standard filenames
test_gs_fnames = id2filenames(test_dir, "gs", test_ids)
