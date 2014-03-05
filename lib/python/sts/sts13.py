"""
define dirs, ids and filenames for STS13 test data
"""

from os.path import join

from sts.io import data_dir, id2filenames

# directory containing original STS 2103 test files
test_dir = join(data_dir, "STS2013-test")

# identifiers for different categories of test data
test_ids = "FNWN", "headlines", "OnWN", "SMT"

# mapping from test identifiers to corresponding input filenames
test_input_fnames = id2filenames(test_dir, "input", test_ids)

# mapping from test identifiers to corresponding gold standard filenames
test_gs_fnames = id2filenames(test_dir, "gs", test_ids)
