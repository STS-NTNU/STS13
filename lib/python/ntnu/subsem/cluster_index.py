from collections import defaultdict
import logging

from operator import itemgetter


# Simple class to hold Brown cluster assignments read from a model file created by wcluster.
class ClusterIndex(object):
    def __init__(self, file, prefix_size=16):
        self.prefix_size = prefix_size
        self.word_to_cluster, self.cluster_to_word, self.freqs = _parse_cluster_file(file, self.prefix_size)

        self.n_cluster = len(self.cluster_to_word.keys())

    def cluster(self, word):
        return self.word_to_cluster[word]

    def cluster_terms(self, cluster, num_terms=10):
        terms = self.cluster_to_word[cluster]
        freqs = [self.freqs[term] for term in terms]
        sorted_terms = sorted(zip(terms, freqs), key=itemgetter(1), reverse=True)

        return [term[0] for term in sorted_terms[0:num_terms]]

    def clusters(self):
        return self.cluster_to_word.keys()


def _parse_cluster_file(f, prefix_size=16):
    line_num = 0
    word_to_cluster = defaultdict(lambda: None)
    cluster_to_word = defaultdict(lambda: [])
    freqs = defaultdict(lambda: 0)

    for line in f.readlines():
        line_num += 1

        tokens = line.strip().split("\t")

        if len(tokens) != 3:
            logging.warn("Couldn't parse line %d" % line_num)
        else:
            c = tokens[0].strip()[0:prefix_size]
            word = tokens[1].strip()
            freq = int(tokens[2])

            if word_to_cluster.has_key(word) or freqs.has_key(word):
                logging.warn("Duplicate entry \"%s\"" % line)

            word_to_cluster[word] = c
            cluster_to_word[c] += [word]
            freqs[word] = freq

    return word_to_cluster, cluster_to_word, freqs