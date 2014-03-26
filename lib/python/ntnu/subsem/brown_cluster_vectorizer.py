from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin
from ntnu.subsem.cluster_index import ClusterIndex


def _build_vocabulary(cluster_index):
    clusters = sorted(cluster_index.clusters())
    vocab = {}

    for i, c in enumerate(clusters):
        vocab[c] = i

    return vocab


# Transforms input into document vectors based on Brown clusters.
# Based on CountVectorizer class in SKLearn.
class BrownClusterVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, cluster_fn, prefix_size=16, preprocessor=None):
        self.cluster_fn = cluster_fn

        self.cluster_index = None
        self.vocabulary_ = None

        self.analyzer = 'word'
        self.preprocessor = preprocessor
        self.strip_accents = 'unicode'
        self.lowercase = True
        self.stop_words = None
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.input = None
        self.ngram_range = (1, 1)
        self.encoding = 'utf-8'
        self.decode_error = 'strict'
        self.prefix_size = prefix_size

        self.analyzer_func = None

    # no data needed, this just reads the model
    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        with open(self.cluster_fn) as f:
            self.cluster_index = ClusterIndex(f, prefix_size=self.prefix_size)

        self.vocabulary_ = _build_vocabulary(self.cluster_index)

        return self

    def transform(self, raw_documents):
        x = lil_matrix((len(raw_documents), self.cluster_index.n_cluster))

        for row, doc in enumerate(raw_documents):
            for token in self.analyzer_func(doc):
                c = self.cluster_index.cluster(token)

                if c:
                    x[row, self.vocabulary_[c]] += 1

        return x.tocsr()
