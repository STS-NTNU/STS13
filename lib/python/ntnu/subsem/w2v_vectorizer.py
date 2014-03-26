from gensim.models import Word2Vec
from numpy import zeros, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin


# Creates document vectors from text based on sublexical Word2Vec log-bilinear models
# trained with Gensim. Creates average of term representations in the document.
# Based on CountVectorizer class in SKLearn.
class Word2VecVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, model_fn, preprocessor=None):
        self.model_fn = model_fn

        self.model = None

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

    # no data needed, this just reads the model
    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        self.model = Word2Vec.load(self.model_fn)

        # pick up external data vectors
        if not hasattr(self.model, 'syn0'):
            self.model.syn0 = load(self.model_fn + '.syn0.npy')

        if not hasattr(self.model, 'syn1'):
            self.model.syn0 = load(self.model_fn + '.syn1.npy')

        return self

    def transform(self, raw_documents):
        x = zeros((len(raw_documents), self.model.layer1_size))

        for row, doc in enumerate(raw_documents):
            n = 0

            for token in self.analyzer_func(doc):
                if token in self.model:
                    x[row, :] += self.model[token]
                    n += 1

            if n > 0:
                x[row, :] = x[row, :] / n

        return x