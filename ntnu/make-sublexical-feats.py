import codecs
import logging
from optparse import OptionParser
import os
import re

from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.spatial.distance import cosine

from ntnu.subsem.brown_cluster_vectorizer import BrownClusterVectorizer
from ntnu.subsem.lsi_vectorizer import LsiVectorizer
from ntnu.subsem.preprocessing import make_preprocessor
from ntnu.subsem.w2v_vectorizer import Word2VecVectorizer
import sts
from sts.io import data_dir


# Creates features based on Sublexical Semantics models. There are two such models:
#
# 1) Character n-gram brown clusters trained with Liu Langs wcluster software. Use the resulting paths file
#    as model file.
# 2) Character n-gram Word2Vec log-bilinear representations trained with the Word2Vec module in Gensim. Models
#    should be saved with Gensims own pickled format.
#
# Features are based on the cosine distance between the document vector generated from each sentence in the sentence
# pair. Document vectors are simple averages of the representations or unweighted cluster frequencies.


STS2014_TEST_ITEMS = [('OnWN', os.path.join(data_dir, 'STS2014-test', 'STS.input.OnWN.txt')),
                      ('deft-forum', os.path.join(data_dir, 'STS2014-test', 'STS.input.deft-forum.txt')),
                      ('deft-news', os.path.join(data_dir, 'STS2014-test', 'STS.input.deft-news.txt')),
                      ('headlines', os.path.join(data_dir, 'STS2014-test', 'STS.input.headlines.txt')),
                      ('images', os.path.join(data_dir, 'STS2014-test', 'STS.input.images.txt')),
                      ('tweet-news', os.path.join(data_dir, 'STS2014-test', 'STS.input.tweet-news.txt'))]


def load_data(f):
    sentences = []

    for line in f:
        if line.strip() == '':
            continue

        line = re.sub('\t+', '\t', line)

        sa, sb = line.split('\t')

        sentences.append((sa.strip(), sb.strip()))

    return sentences


def generate_sublexical_brown_features(in_fn, out_path, data_id, fitted_vect, order, model_id):
    data_dir= data_id[9:] if data_id.startswith("surprise.") else data_id
    out_dir = os.path.join(out_path, data_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    preprocess = make_preprocessor(order=order)

    feat_fn = os.path.join(out_dir, model_id + '.txt')
    logging.info('Writing features to %s' % feat_fn)

    with open(feat_fn, 'w') as out_f:
        with codecs.open(in_fn, 'r', 'utf-8') as in_f:
            for idx, (sa, sb) in enumerate(load_data(in_f)):
                sa_vec = fitted_vect.transform([preprocess(sa)])

                if issparse(sa_vec):
                    sa_vec = sa_vec.todense()

                sb_vec = fitted_vect.transform([preprocess(sb)])

                if issparse(sb_vec):
                    sb_vec = sb_vec.todense()

                if norm(sa_vec) == 0 or norm(sb_vec) == 0:
                    logging.warn("zero norm doc vector in %s" % data_id)
                    dist = 1.0
                else:
                    dist = cosine(sa_vec, sb_vec)

                out_f.write("%f\n" % dist)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option('-n', '--ngram-order')
    parser.add_option('-m', '--model-file')
    parser.add_option('-i', '--model-id')
    parser.add_option('-o', '--out-path', default='out')
    parser.add_option('-t', '--model-type', default='brown')

    opts, args = parser.parse_args()

    if opts.ngram_order:
        order = int(opts.ngram_order)
    else:
        raise ValueError('--order argument is required')

    if opts.model_file:
        model_fn = opts.model_file
    else:
        raise ValueError('--model-file argument is required')

    if opts.model_id:
        model_id = opts.model_id
    else:
        model_id = "ntnu.subsem-%d" % order

    out_path = opts.out_path

    if opts.model_type == 'brown':
        vectorizer = BrownClusterVectorizer(model_fn).fit([])
    elif opts.model_type == 'w2v':
        vectorizer = Word2VecVectorizer(model_fn).fit([])
    elif opts.model_type == 'lsi':
        vectorizer = LsiVectorizer(model_fn).fit([])
    else:
        raise ValueError('invalid --model-type')

    # STS2012 train
    for data_id, input_fn in sts.sts12.train_input_fnames.items():
        logging.info("Creating features for STS2012 train %s" % data_id)
        generate_sublexical_brown_features(input_fn, os.path.join(out_path, 'STS2012-train'),
                                           data_id, vectorizer, order, model_id)

    # STS2012 test
    for data_id, input_fn in sts.sts12.test_input_fnames.items():
        logging.info("Creating features for STS2012 test %s" % data_id)
        generate_sublexical_brown_features(input_fn, os.path.join(out_path, 'STS2012-test'),
                                           data_id, vectorizer, order, model_id)

    # STS2013 test
    for data_id, input_fn in sts.sts13.test_input_fnames.items():
        logging.info("Creating features for STS2013 test %s" % data_id)
        generate_sublexical_brown_features(input_fn, os.path.join(out_path, 'STS2013-test'),
                                           data_id, vectorizer, order, model_id)

    # STS2014 trial
    for data_id, input_fn in sts.sts14.trial_input_fnames.items():
        logging.info("Creating features for STS2014 trial %s" % data_id)
        generate_sublexical_brown_features(input_fn, os.path.join(out_path, 'STS2014-trial'),
                                           data_id, vectorizer, order, model_id)
    # STS2014 test
    for data_id, input_fn in STS2014_TEST_ITEMS:
        logging.info("Creating features for STS2014 trial %s" % data_id)
        generate_sublexical_brown_features(input_fn, os.path.join(out_path, 'STS2014-test'),
                                           data_id, vectorizer, order, model_id)
