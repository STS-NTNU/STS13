import logging

from numpy import mean
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

from ntnu.featcluster.cluster import DatasetRelevanceClassifier
from ntnu.featcluster.data import read_data
from ntnu.feats import takelab_feats, takelab_lsa_feats
from ntnu.io import feat_dir
from sts import sts14
from sts.score import correlation

# Dataset relevance search using clustering.

# Parameter search results are cool, but held out results are disappointing.

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

# Results with Takelab and subsem-lsitfidf-wiki8-n4-c2000-min5-raw-cos features. No param grid.

# Mean	FNWN	OnWN	SMT	headlines
# 0.5770	0.3565	0.4679	0.7078	0.7756	k-means-1-medoid-top
# 0.5637	0.3548	0.4172	0.7065	0.7763	k-means-1-medoid-max_ratio_gap
# 0.5495	0.3565	0.4679	0.6658	0.7078	k-means-1-centroid-top
# 0.5486	0.3505	0.4172	0.7086	0.7180	k-means-1-centroid-max_ratio_gap
# 0.5564	0.3565	0.4590	0.7023	0.7078	k-means-3-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-3-medoid-max_ratio_gap
# 0.5495	0.3565	0.4679	0.6658	0.7078	k-means-3-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-3-centroid-max_ratio_gap
# 0.5757	0.3603	0.4590	0.7078	0.7756	k-means-5-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-5-medoid-max_ratio_gap
# 0.5596	0.3603	0.4679	0.7023	0.7078	k-means-5-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-5-centroid-max_ratio_gap
# 0.5619	0.3603	0.4679	0.7078	0.7116	k-means-9-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-9-medoid-max_ratio_gap
# 0.5485	0.3840	0.4679	0.5663	0.7756	k-means-9-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-9-centroid-max_ratio_gap
# 0.5151	0.3603	0.4679	0.5663	0.6658	k-means-15-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-15-medoid-max_ratio_gap
# 0.5770	0.3565	0.4679	0.7078	0.7756	k-means-15-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	k-means-15-centroid-max_ratio_gap
# 0.5252	0.3430	0.3921	0.6538	0.7121	k-means-3-medoid-top-n
# 0.5282	0.3548	0.3921	0.6538	0.7121	k-means-3-centroid-top-n
# 0.5614	0.3586	0.3921	0.7186	0.7763	k-means-5-medoid-top-n
# 0.5617	0.3549	0.3921	0.7186	0.7813	k-means-5-centroid-top-n
# 0.5668	0.3501	0.4172	0.7186	0.7813	k-means-9-medoid-top-n
# 0.5730	0.3549	0.4372	0.7186	0.7813	k-means-9-centroid-top-n
# 0.5638	0.3501	0.4172	0.7065	0.7813	k-means-15-medoid-top-n
# 0.5717	0.3501	0.4372	0.7180	0.7813	k-means-15-centroid-top-n
# 0.5816	0.3840	0.4590	0.7078	0.7756	agglomerative-3-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-3-medoid-max_ratio_gap
# 0.5633	0.3840	0.4590	0.7023	0.7078	agglomerative-3-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-3-centroid-max_ratio_gap
# 0.5393	0.3565	0.4590	0.5663	0.7756	agglomerative-5-medoid-top
# 0.5564	0.3501	0.3921	0.7021	0.7813	agglomerative-5-medoid-max_ratio_gap
# 0.5220	0.3603	0.4590	0.5663	0.7023	agglomerative-5-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-5-centroid-max_ratio_gap
# 0.5444	0.3448	0.4590	0.6658	0.7078	agglomerative-9-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-9-medoid-max_ratio_gap
# 0.5747	0.3565	0.4590	0.7078	0.7756	agglomerative-9-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-9-centroid-max_ratio_gap
# 0.5279	0.3840	0.4590	0.5663	0.7023	agglomerative-15-medoid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-15-medoid-max_ratio_gap
# 0.5656	0.3840	0.4590	0.7078	0.7116	agglomerative-15-centroid-top
# 0.5677	0.3501	0.4372	0.7021	0.7813	agglomerative-15-centroid-max_ratio_gap
# 0.5493	0.3581	0.3921	0.7180	0.7291	agglomerative-3-medoid-top-n
# 0.5340	0.3720	0.3921	0.6538	0.7180	agglomerative-3-centroid-top-n
# 0.5582	0.3549	0.3921	0.7096	0.7763	agglomerative-5-medoid-top-n
# 0.5493	0.3549	0.3921	0.7186	0.7317	agglomerative-5-centroid-top-n
# 0.5285	0.3501	0.3921	0.6538	0.7180	agglomerative-9-medoid-top-n
# 0.5605	0.3549	0.3921	0.7186	0.7763	agglomerative-9-centroid-top-n
# 0.5717	0.3501	0.4372	0.7180	0.7813	agglomerative-15-medoid-top-n
# 0.5667	0.3501	0.4172	0.7180	0.7813	agglomerative-15-centroid-top-n

# Held out evaluation scores
# OnWN	0.6459
# headlines	0.7194
# FNWN	0.2620
# SMT	0.3646
# Mean	0.4979
# Data set predictions for STS2014-test
#     deft-news	['surprise.OnWN']
# deft-forum	['MSRpar']
# OnWN	['MSRvid']
# tweet-news	['MSRpar']
# images	['MSRvid']
# headlines	['surprise.SMTnews']


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
        x_test = frame[frame.data_set == 'STS2013-test']
        x_test = x_test[x_test.data_id == data_id][feats].as_matrix()
        y_test = frame[frame.data_set == 'STS2013-test']
        y_test = y_test[y_test.data_id == data_id]['gs'].values

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

    feats = FEATS + ['subsem-lsitfidf-wiki8-n4-c2000-min5-raw-cos']
    df = read_data(feats, feat_dir)

    # print "Mean\t%s\t%s\t%s\t%s" % tuple(sorted(ALL_TEST_IDS))
    #
    # for exp in EXPERIMENTS:
    #     exp_descr = '-'.join([str(val[1]) for val in sorted(exp.items(), key=itemgetter(0))])
    #
    #     result = run_experiment(df, ALL_TEST_IDS, svm_grid=None, feats=feats, **exp)
    #
    #     result = sorted(result, key=itemgetter(1))
    #     mean_score = mean([x[1] for x in result])
    #
    #     print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % tuple([mean_score] + [x[1] for x in result] + [exp_descr])

    # Held out eval
    print "Held out evaluation scores"
    keyed_train_data = {}

    for data_id in ALL_TRAIN_IDS:
        m = df[df.data_id == data_id][feats].as_matrix()
        keyed_train_data[data_id] = m

    keyed_test_data = {}

    for data_id in ALL_TEST_IDS:
        m = df[df.data_id == data_id][feats].as_matrix()
        keyed_test_data[data_id] = m

    drc = DatasetRelevanceClassifier(method='agglomerative', selection='top', representative='medoid', n_clusters=3)
    drc.fit(keyed_train_data)

    scores = []

    for data_id, X in keyed_test_data.items():
        train_sets = drc.predict(X)

        x_train = df[df.data_id.isin(train_sets)][feats].as_matrix()
        y_train = df[df.data_id.isin(train_sets)]['gs'].values
        x_test = df[df.data_set == 'STS2013-test']
        x_test = x_test[x_test.data_id == data_id][feats].as_matrix()
        y_test = df[df.data_set == 'STS2013-test']
        y_test = y_test[y_test.data_id == data_id]['gs'].values

        # use second half of test sets for final evaluation
        x_test = x_test[len(y_test)/2:, :]
        y_test = y_test[len(y_test)/2:]

        model = SVR()
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        score = correlation(y_test, pred)

        scores.append(score)

        print "%s\t%.04f" % (data_id, score)

    print "Mean\t%.04f" % mean(scores)

    # 2014 predictions
    print "Data set predictions for STS2014-test"

    keyed_train_data = {}

    for data_id in ALL_TRAIN_IDS:
        m = df[df.data_id == data_id][feats].as_matrix()
        keyed_train_data[data_id] = m

    keyed_test_data = {}

    for data_id in sts14.test_ids:
        m = df[df.data_set == 'STS2014-test']
        m = df[df.data_id == data_id][feats].as_matrix()
        keyed_test_data[data_id] = m

    drc = DatasetRelevanceClassifier(method='k-means', selection='top', representative='medoid', n_clusters=9)
    drc.fit(keyed_train_data)

    for data_id, X in keyed_test_data.items():
        train_sets = drc.predict(X)

        print "%s\t%s" % (data_id, train_sets)