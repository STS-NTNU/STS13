"""
define dirs, ids and filenames for STS13 test data
"""

from os.path import join, normpath

def id2filenames(dir, type, ids):
    return {id: join(dir, "STS.{}.{}.txt".format(type, id)) 
            for id in ids}

# find root of Git repos
repos_dir = normpath(join(__file__, "../../../.."))


test_dir = join(repos_dir, "data/STS2013-test")
test_ids = "FNWN", "headlines", "OnWN", "SMT"
test_input_fnames = id2filenames(test_dir, "input", test_ids)
test_gs_fnames = id2filenames(test_dir, "gs", test_ids)
