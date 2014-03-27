import logging
from operator import itemgetter

from numpy import mean
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

from ntnu.featcluster.cluster import DatasetRelevanceClassifier
from ntnu.featcluster.data import read_data
from ntnu.feats import takelab_feats, takelab_lsa_feats
from ntnu.io import feat_dir
from sts.score import correlation


# Parameter search for the DatasetRelevanceClassifier

# Results run with Takelab features and no parameter grid

# Mean	FNWN	OnWN	SMT	headlines
# 0.5645	0.3322	0.4651	0.7038	0.7568	k-means-1-medoid-top
# 0.5530	0.3355	0.4133	0.6945	0.7686	k-means-1-medoid-max_ratio_gap
# 0.5325	0.3322	0.4651	0.6290	0.7038	k-means-1-centroid-top
# 0.5581	0.3374	0.4133	0.7130	0.7686	k-means-1-centroid-max_ratio_gap
# 0.5032	0.3322	0.4651	0.5443	0.6714	k-means-3-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-3-medoid-max_ratio_gap
# 0.4937	0.3366	0.4651	0.5443	0.6290	k-means-3-centroid-top
# 0.5579	0.3297	0.4393	0.6911	0.7716	k-means-3-centroid-max_ratio_gap
# 0.5455	0.3503	0.4567	0.6714	0.7038	k-means-5-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-5-medoid-max_ratio_gap
# 0.5476	0.3503	0.4651	0.6714	0.7038	k-means-5-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-5-centroid-max_ratio_gap
# 0.5734	0.3681	0.4651	0.7038	0.7568	k-means-9-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-9-medoid-max_ratio_gap
# 0.4972	0.3503	0.4651	0.5443	0.6290	k-means-9-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-9-centroid-max_ratio_gap
# 0.5521	0.3681	0.4651	0.6714	0.7038	k-means-15-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-15-medoid-max_ratio_gap
# 0.5410	0.3322	0.4567	0.6714	0.7038	k-means-15-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	k-means-15-centroid-max_ratio_gap
# 0.5102	0.3159	0.3891	0.6227	0.7130	k-means-3-medoid-top-n
# 0.5102	0.3159	0.3891	0.6227	0.7130	k-means-3-centroid-top-n
# 0.5516	0.3355	0.3891	0.7134	0.7686	k-means-5-medoid-top-n
# 0.5517	0.3327	0.3891	0.7134	0.7716	k-means-5-centroid-top-n
# 0.5570	0.3327	0.4133	0.7134	0.7686	k-means-9-medoid-top-n
# 0.5428	0.3327	0.4393	0.6863	0.7130	k-means-9-centroid-top-n
# 0.5523	0.3297	0.4133	0.6945	0.7716	k-means-15-medoid-top-n
# 0.5609	0.3327	0.4393	0.7000	0.7716	k-means-15-centroid-top-n
# 0.5325	0.3322	0.4651	0.6290	0.7038	agglomerative-3-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-3-medoid-max_ratio_gap
# 0.5336	0.3366	0.4651	0.6290	0.7038	agglomerative-3-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-3-centroid-max_ratio_gap
# 0.5291	0.3503	0.4651	0.5443	0.7568	agglomerative-5-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-5-medoid-max_ratio_gap
# 0.5455	0.3503	0.4567	0.6714	0.7038	agglomerative-5-centroid-top
# 0.5579	0.3297	0.4393	0.6911	0.7716	agglomerative-5-centroid-max_ratio_gap
# 0.5501	0.3322	0.4651	0.6995	0.7038	agglomerative-9-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-9-medoid-max_ratio_gap
# 0.5547	0.3503	0.4651	0.6995	0.7038	agglomerative-9-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-9-centroid-max_ratio_gap
# 0.5500	0.3681	0.4567	0.6714	0.7038	agglomerative-15-medoid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-15-medoid-max_ratio_gap
# 0.5500	0.3681	0.4567	0.6714	0.7038	agglomerative-15-centroid-top
# 0.5581	0.3297	0.4399	0.6911	0.7716	agglomerative-15-centroid-max_ratio_gap
# 0.5071	0.3159	0.3891	0.6227	0.7006	agglomerative-3-medoid-top-n
# 0.5156	0.3374	0.3891	0.6227	0.7134	agglomerative-3-centroid-top-n
# 0.5169	0.3424	0.3891	0.6227	0.7134	agglomerative-5-medoid-top-n
# 0.5525	0.3391	0.3891	0.7134	0.7686	agglomerative-5-centroid-top-n
# 0.5327	0.3423	0.3891	0.6863	0.7130	agglomerative-9-medoid-top-n
# 0.5577	0.3327	0.4133	0.7130	0.7716	agglomerative-9-centroid-top-n
# 0.5530	0.3327	0.4133	0.6945	0.7716	agglomerative-15-medoid-top-n
# 0.5522	0.3327	0.4133	0.6911	0.7716	agglomerative-15-centroid-top-n


FEATS = takelab_feats + takelab_lsa_feats
ALL_TRAIN_IDS = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.SMTnews', 'surprise.OnWN']
ALL_TEST_IDS = ['FNWN', 'headlines', 'OnWN', 'SMT']

LARGE_PARAM_GRID = {
    'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    'epsilon': [0.01, 0.03, 0.1, 0.3, 1, 3],
    'gamma': [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
}

PARAM_GRID = {
    'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    'epsilon': [0.03, 0.1, 0.3, 1],
    'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
}

SMALL_PARAM_GRID = {
    'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1]
}

FIXED_C_GRID = {
    'C': [1],
    'epsilon': [0.1, 0.3, 1, 3],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1]
}


def run_experiment(frame, test_ids, train_ids=ALL_TRAIN_IDS, feats=FEATS, svm_grid=SMALL_PARAM_GRID,
                   cv=5, n_jobs=1,**drc_args):
    keyed_train_data = {}

    for data_id in train_ids:
        m = frame[frame.data_id == data_id][feats].as_matrix()
        keyed_train_data[data_id] = m

    keyed_test_data = {}

    for data_id in test_ids:
        m = frame[frame.data_id == data_id][feats].as_matrix()
        keyed_test_data[data_id] = m

    drc = DatasetRelevanceClassifier(**drc_args)
    drc.fit(keyed_train_data)

    result = []

    for data_id, X in keyed_test_data.items():
        train_sets = drc.predict(X)

        x_train = frame[frame.data_id.isin(train_sets)][feats].as_matrix()
        y_train = frame[frame.data_id.isin(train_sets)]['gs'].values
        x_test = frame[frame.data_id == data_id][feats].as_matrix()
        y_test = frame[frame.data_id == data_id]['gs'].values

        # reserve second half of test sets for final evaluation
        x_test = x_test[0:len(y_test)/2, :]
        y_test = y_test[0:len(y_test)/2]

        if svm_grid:
            grid = GridSearchCV(SVR(), svm_grid, cv=cv, verbose=1, n_jobs=n_jobs)
            grid.fit(x_train, y_train)

            best_params = grid.best_params_
        else:
            best_params = {}

        model = SVR(**best_params)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        score = correlation(y_test, pred)

        result.append((data_id, score, train_sets, best_params))

    return result

EXPERIMENTS = [
    {'method': 'k-means', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 1},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 1},
    {'method': 'k-means', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 1},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 1},

    {'method': 'k-means', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'k-means', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 3},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 3},

    {'method': 'k-means', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'k-means', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 5},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 5},

    {'method': 'k-means', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'k-means', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 9},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 9},

    {'method': 'k-means', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'k-means', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 15},
    {'method': 'k-means', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 15},

    {'method': 'k-means', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 3},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 5},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 9},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'k-means', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 15},

    {'method': 'agglomerative', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'agglomerative', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 3},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 3},

    {'method': 'agglomerative', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'agglomerative', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 5},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 5},

    {'method': 'agglomerative', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'agglomerative', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 9},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 9},

    {'method': 'agglomerative', 'selection': 'top', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'agglomerative', 'selection': 'top', 'representative': 'centroid', 'n_clusters': 15},
    {'method': 'agglomerative', 'selection': 'max_ratio_gap', 'representative': 'centroid', 'n_clusters': 15},

    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 3},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 3},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 5},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 5},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 9},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 9},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'medoid', 'n_clusters': 15},
    {'method': 'agglomerative', 'selection': 'top-n', 'representative': 'centroid', 'n_clusters': 15}
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    df = read_data(FEATS, feat_dir)

    print "Mean\t%s\t%s\t%s\t%s" % tuple(sorted(ALL_TEST_IDS))

    for exp in EXPERIMENTS:
        exp_descr = '-'.join([str(val[1]) for val in sorted(exp.items(), key=itemgetter(0))])

        result = run_experiment(df, ALL_TEST_IDS, svm_grid=None, **exp)

        result = sorted(result, key=itemgetter(1))
        mean_score = mean([x[1] for x in result])

        print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % tuple([mean_score] + [x[1] for x in result] + [exp_descr])