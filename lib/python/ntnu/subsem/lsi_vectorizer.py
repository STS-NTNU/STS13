import os
from gensim.models import LsiModel, TfidfModel
from numpy import zeros
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin


class LsiVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, model_fn, preprocessor=None):
        self.model_fn = model_fn

        self.model = None
        self.tfidf = None

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

        self.analyzer_func = None

    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        self.model = LsiModel.load(self.model_fn)

        if os.path.exists(self.model_fn + '.tfidf'):
            self.tfidf = TfidfModel.load(self.model_fn + '.tfidf')

        return self

    def transform(self, raw_documents):
        x = zeros((len(raw_documents), self.model.num_topics))

        for row, doc in enumerate(raw_documents):
            doc = self.model.id2word.doc2bow(self.analyzer_func(doc))

            if self.tfidf:
                topics = self.model[self.tfidf[doc]]
            else:
                topics = self.model[doc]

            for idx, val in topics:
                x[row, idx] = val

        return x