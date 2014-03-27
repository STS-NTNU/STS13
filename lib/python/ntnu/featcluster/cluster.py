import logging
from operator import itemgetter

from numpy import array, nditer, unique, zeros, sum
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, Ward


# DataSetRelevanceClassifer attempts to identify the most effective subsets of the training set. This is done
# by matching the clusters of the test set to training set clusters. The clusters are based on unsupervised k-means
# or agglomerative clustering.
#
# Run on the 2013 data with results are found in ntnu/cluster-features.py.


def _create_cluster(m, representative):
    centroid = m.sum(axis=0)/m.shape[0]

    if representative == 'centroid':
        return centroid
    elif representative == 'medoid':
        centroid.shape = 1, len(centroid)
        centroid_dists = cdist(m, centroid, 'euclidean')
        repr_idx = centroid_dists.argmin()

        return m[repr_idx, :]
    else:
        raise NotImplementedError


def _select_clusters(keyed_dists):
    keys = [x[0] for x in keyed_dists]
    dists = array([x[1] for x in keyed_dists])

    dist_ratios = (dists[0:-1]- dists[1:])/dists[1:]

    return keys[0:dist_ratios.argmax() + 1]


def _representative(centroids, X, representative):
    if representative == 'centroid':
        return centroids
    elif representative == 'medoid':
        centroid_dists = cdist(X, centroids, 'euclidean')
        repr_idxs = centroid_dists.argmin(axis=0)

        return X[repr_idxs, :]

def _create_kmeans_clusters(X, model, representative):
    model.fit(X)

    centroids = model.cluster_centers_

    return _representative(centroids, X, representative)


def _create_agglo_clusters(X, model, representative):
    y = model.fit_predict(X)
    p = X.shape[1]
    n = model.n_clusters

    centroids = zeros((n, p))

    for c in unique(y):
        c_x = X[y == c, :]

        centroids[c, :] = sum(c_x, axis=0)/c_x.shape[0]

    return _representative(centroids, X, representative)


def _get_cluster_func(method, n_clusters, representative):
    if n_clusters == 1:
        return lambda X: [_create_cluster(X, representative)]
    elif method == 'k-means':
        return lambda X: _create_kmeans_clusters(X, KMeans(n_clusters=n_clusters), representative)
    elif method == 'agglomerative':
        return lambda X: _create_agglo_clusters(X, Ward(n_clusters=n_clusters), representative)
    else:
        raise NotImplementedError


def _top_clusters(keyed_dists, n):
    keys = [x[0] for x in keyed_dists]

    return unique(keys[0:n])


def _get_select_func(selection, n_clusters):
    if selection == 'max_ratio_gap':
        return _select_clusters
    elif selection == 'top':
        return lambda keyed_dists: _top_clusters(keyed_dists, 1)
    elif selection == 'top-n':
        return lambda keyed_dists: _top_clusters(keyed_dists, n_clusters)
    else:
        raise NotImplementedError


class DatasetRelevanceClassifier(object):
    def __init__(self, method='k-means', n_clusters=1, selection='max_ratio_gap', representative='centroid'):
        self.method = method
        self.n_clusters = n_clusters
        self.selection = selection

        self.clusters = None
        self.cluster_func = _get_cluster_func(method, n_clusters, representative)
        self.select_func = _get_select_func(selection, n_clusters)

    def fit(self, keyed_datasets):
        clusters = []

        for data_id, X in keyed_datasets.items():
            clusters.append((data_id, self.cluster_func(X)))

        self.clusters = clusters

        return self

    def predict(self, X):
        clusters = self.cluster_func(X)

        dists = []

        for data_id, c in self.clusters:
            d = cdist(clusters, c, 'euclidean')

            for x in nditer(d):
                dists.append((data_id, float(x)))

        dists = sorted(dists, key=itemgetter(1))

        logging.debug("%s: computed distances %s" % (self.__class__.__name__, dists))

        return self.select_func(dists)
