"""
define dirs, ids and filenames for STS12 training and testing data
"""

from os.path import join, normpath

def filenames(dir, type, ids):
    return [join(dir, "STS.{}.{}.txt".format(type, id)) for id in ids]

# find root of Git repos
repos_dir = normpath(join(__file__, "../../../.."))

train_dir = join(repos_dir, "data/STS2012-test")
train_ids = "MSRpar", "MSRvid", "SMTeuroparl"
train_input_fnames = filenames(train_dir, "input", train_ids)
train_gs_fnames = filenames(train_dir, "gs", train_ids)

test_dir = join(repos_dir, "data/STS2012-test")
test_ids = "MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"
test_input_fnames = filenames(test_dir, "input", test_ids)
test_gs_fnames = filenames(test_dir, "gs", test_ids)

