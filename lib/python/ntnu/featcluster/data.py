import os
from numpy import loadtxt
from pandas import Series, DataFrame
import sts
from sts.io import repos_dir


# Reads feature data into a pandas dataframe.
# Used by feature-clustering.py script


data_paths_and_ids = [('STS2012-train', sts.sts12.train_ids),
                       ('STS2012-test', sts.sts12.test_ids),
                       ('STS2013-test', sts.sts13.test_ids),
                       ('STS2014-test', sts.sts14.test_ids)]


def series_from_feat(feat_fn):
    vals = []

    with open(feat_fn) as f:
        for line in f:
            vals.append(float(line.strip()))

    return Series(vals)


def frame_for_id(features, feat_path='out', data_ids=sts.sts12.train_ids, data_dir='STS2012-train'):
    frame = DataFrame()

    for data_id in data_ids:
        data = {}

        for feat_id in features:
            data_id_dir = data_id[9:] if data_id.startswith("surprise.") else data_id
            feat_fn = os.path.join(feat_path, data_dir, data_id_dir, "%s.txt" % feat_id)

            data[feat_id] = series_from_feat(feat_fn)

        new_frame = DataFrame(data)
        new_frame['data_id'] = data_id

        gs_fn = os.path.join(repos_dir, 'data', data_dir, "STS.gs.%s.txt" % data_id)

        if os.path.exists(gs_fn):
            new_frame['gs'] = Series(loadtxt(gs_fn))
        else:
            new_frame['gs'] = None

        frame = frame.append(new_frame)

    frame['data_set'] = data_dir

    return frame


def read_data(features, feat_path='out'):
    frame = DataFrame()

    for data_path, data_ids in data_paths_and_ids:
        frame = frame.append(frame_for_id(features, feat_path, data_ids, data_path))

    return frame

