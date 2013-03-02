#!/usr/bin/env python
# coding=utf-8

from collections import Counter, defaultdict
import math
import re
import sys

import numpy as np
from numpy.linalg import norm

import nltk
from nltk.corpus import wordnet


stopwords = set([
"i", "a", "about", "an", "are", "as", "at", "be", "by", "for", "from",
"how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
"was", "what", "when", "where", "who", "will", "with", "the", "'s", "did",
"have", "has", "had", "were", "'ll"
])


to_wordnet_tag = {
        'NN':wordnet.NOUN,
        'JJ':wordnet.ADJ,
        'VB':wordnet.VERB,
        'RB':wordnet.ADV
    }


def load_wweight_table(path):
    lines = open(path).readlines()
    wweight = defaultdict(float)
    if not len(lines):
        return (wweight, 0.)
    totfreq = int(lines[0])
    for l in lines[1:]:
        w, freq = l.split()
        freq = float(freq)
        if freq < 10:
            continue
        wweight[w] = math.log(totfreq / freq)

    return wweight

def load_data(path):
    sentences_pos = []
    r1 = re.compile(r'\<([^ ]+)\>')
    r2 = re.compile(r'\$US(\d)')
    for l in open(path):
        l = l.decode('utf-8')
        l = l.replace(u'’', "'")
        l = l.replace(u'``', '"')
        l = l.replace(u"''", '"')
        l = l.replace(u"—", '--')
        l = l.replace(u"–", '--')
        l = l.replace(u"´", "'")
        l = l.replace(u"-", " ")
        l = l.replace(u"/", " ")
        l = r1.sub(r'\1', l)
        l = r2.sub(r'$\1', l)
        s = l.strip().split('\t')
        sa, sb = tuple(nltk.word_tokenize(s)
                          for s in l.strip().split('\t'))
        sa, sb = ([x.encode('utf-8') for x in sa],
                  [x.encode('utf-8') for x in sb])

        for s in (sa, sb):
            for i in xrange(len(s)):
                if s[i] == "n't":
                    s[i] = "not"
                elif s[i] == "'m":
                    s[i] = "am"
        sa, sb = fix_compounds(sa, sb), fix_compounds(sb, sa)
        sentences_pos.append((nltk.pos_tag(sa), nltk.pos_tag(sb)))
    return sentences_pos


def fix_compounds(a, b):
    sb = set(x.lower() for x in b)

    a_fix = []
    la = len(a)
    i = 0
    while i < la:
        if i + 1 < la:
            comb = a[i] + a[i + 1]
            if comb.lower() in sb:
                a_fix.append(a[i] + a[i + 1])
                i += 2
                continue
        a_fix.append(a[i])
        i += 1
    return a_fix


word_matcher = re.compile('[^0-9,.(=)\[\]/_`]+$')

def is_word(w):
    return word_matcher.match(w) is not None


def get_locase_words(spos):
    return [x[0].lower() for x in spos
            if is_word(x[0])]


def get_lemmatized_words(sa):
    rez = []
    for w, wpos in sa:
        w = w.lower()
        if w in stopwords or not is_word(w):
            continue
        wtag = to_wordnet_tag.get(wpos[:2])
        if wtag is None:
            wlem = w
        else:
            wlem = wordnet.morphy(w, wtag) or w
        rez.append(wlem)
    return rez


def is_stock_tick(w):
    return w[0] == '.' and len(w) > 1 and w[1:].isupper()


def stocks_matches(sa, sb):
    ca = set(x[0] for x in sa if is_stock_tick(x[0]))
    cb = set(x[0] for x in sb if is_stock_tick(x[0]))
    isect = len(ca.intersection(cb))
    la = len(ca)
    lb = len(cb)

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f)


def case_matches(sa, sb):
    ca = set(x[0] for x in sa[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    cb = set(x[0] for x in sb[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    la = len(ca)
    lb = len(cb)
    isect = len(ca.intersection(cb))

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f)


risnum = re.compile(r'^[0-9,./-]+$')
rhasdigit = re.compile(r'[0-9]')


def match_number(xa, xb):
    if xa == xb:
        return True
    xa = xa.replace(',', '')
    xb = xb.replace(',', '')

    try:
        va = int(float(xa))
        vb = int(float(xb))
        if (va == 0 or vb == 0) and va != vb:
            return False
        fxa = float(xa)
        fxb = float(xb)
        if abs(fxa - fxb) > 1:
            return False
        diga = xa.find('.')
        digb = xb.find('.')
        diga = 0 if diga == -1 else len(xa) - diga - 1
        digb = 0 if digb == -1 else len(xb) - digb - 1
        if diga > 0 and digb > 0 and va != vb:
            return False
        dmin = min(diga, digb)
        if dmin == 0:
            if abs(round(fxa, 0) - round(fxb, 0)) < 1e-5:
                return True
            return va == vb
        return abs(round(fxa, dmin) - round(fxb, dmin)) < 1e-5
    except:
        pass

    return False


def number_features(sa, sb):
    numa = set(x[0] for x in sa if risnum.match(x[0]) and
            rhasdigit.match(x[0]))
    numb = set(x[0] for x in sb if risnum.match(x[0]) and
            rhasdigit.match(x[0]))
    isect = 0
    for na in numa:
        if na in numb:
            isect += 1
            continue
        for nb in numb:
            if match_number(na, nb):
                isect += 1
                break

    la, lb = len(numa), len(numb)

    f = 1.
    subset = 0.
    if la + lb > 0:
        if isect == la or isect == lb:
            subset = 1.
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2. * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f, subset)


def len_compress(l):
    return math.log(1. + l)


def make_ngrams(l, n):
    rez = [l[i:(-n + i + 1)] for i in xrange(n - 1)]
    rez.append(l[n - 1:])
    return zip(*rez)


def ngram_match(sa, sb, n):
    nga = make_ngrams(sa, n)
    ngb = make_ngrams(sb, n)
    matches = 0
    c1 = Counter(nga)
    for ng in ngb:
        if c1[ng] > 0:
            c1[ng] -= 1
            matches += 1
    p = 0.
    r = 0.
    f1 = 1.
    if len(nga) > 0 and len(ngb) > 0:
        p = matches / float(len(nga))
        r = matches / float(len(ngb))
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1


wpathsimcache = {}
def wpathsim(a, b):
    if a > b:
        b, a = a, b
    p = (a, b)
    if p in wpathsimcache:
        return wpathsimcache[p]
    if a == b:
        wpathsimcache[p] = 1.
        return 1.
    sa = wordnet.synsets(a)
    sb = wordnet.synsets(b)
    mx = max([wa.path_similarity(wb)
              for wa in sa
              for wb in sb
              ] + [0.])
    wpathsimcache[p] = mx
    return mx


def calc_wn_prec(lema, lemb):
    rez = 0.
    for a in lema:
        ms = 0.
        for b in lemb:
            ms = max(ms, wpathsim(a, b))
        rez += ms
    return rez / len(lema)

def wn_sim_match(lema, lemb):
    f1 = 1.
    p = 0.
    r = 0.
    if len(lema) > 0 and len(lemb) > 0:
        p = calc_wn_prec(lema, lemb)
        r = calc_wn_prec(lemb, lema)
        f1 = 2. * p * r / (p + r) if p + r > 0 else 0.
    return f1


def relative_len_difference(lca, lcb):
    la, lb = len(lca), len(lcb)
    return abs(la - lb) / float(max(la, lb) + 1e-5)


def relative_ic_difference(lca, lcb):
    #wa = sum(wweight[x] for x in lca)
    #wb = sum(wweight[x] for x in lcb)
    wa = sum(max(0., wweight[x] - minwweight) for x in lca)
    wb = sum(max(0., wweight[x] - minwweight) for x in lcb)
    return abs(wa - wb) / (max(wa, wb) + 1e-5)


def weighted_word_match(lca, lcb):
    wa = Counter(lca)
    wb = Counter(lcb)
    wsuma = sum(wweight[w] * wa[w] for w in wa)
    wsumb = sum(wweight[w] * wb[w] for w in wb)
    wsum = 0.

    for w in wa:
        wd = min(wa[w], wb[w])
        wsum += wweight[w] * wd
    p = 0.
    r = 0.
    if wsuma > 0 and wsum > 0:
        p = wsum / wsuma
    if wsumb > 0 and wsum > 0:
        r = wsum / wsumb
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1



def calc_features(sa, sb, with_lsa):
    olca = get_locase_words(sa)
    olcb = get_locase_words(sb)
    lca = [w for w in olca if w not in stopwords]
    lcb = [w for w in olcb if w not in stopwords]
    lema = get_lemmatized_words(sa)
    lemb = get_lemmatized_words(sb)

    f = []
    f += number_features(sa, sb)
    f += case_matches(sa, sb)
    f += stocks_matches(sa, sb)
    f += [
            ngram_match(lca, lcb, 1),
            ngram_match(lca, lcb, 2),
            ngram_match(lca, lcb, 3),
            ngram_match(lema, lemb, 1),
            ngram_match(lema, lemb, 2),
            ngram_match(lema, lemb, 3),
            wn_sim_match(lema, lemb),
            weighted_word_match(olca, olcb),
            weighted_word_match(lema, lemb),
            relative_len_difference(lca, lcb),
            relative_ic_difference(olca, olcb)
        ]
    
    if with_lsa:
        f += [
            dist_sim(nyt_sim, lema, lemb),
            #dist_sim(wiki_sim, lema, lemb),
            weighted_dist_sim(nyt_sim, lema, lemb),
            weighted_dist_sim(wiki_sim, lema, lemb)
        ]

    return f


        
class Sim:
    def __init__(self, words, vectors):
        self.word_to_idx = {a: b for b, a in
                            enumerate(w.strip() for w in open(words))}
        self.mat = np.loadtxt(vectors)

    def bow_vec(self, b):
        vec = np.zeros(self.mat.shape[1])
        for k, v in b.iteritems():
            idx = self.word_to_idx.get(k, -1)
            if idx >= 0:
                vec += self.mat[idx] / (norm(self.mat[idx]) + 1e-8) * v
        return vec

    def calc(self, b1, b2):
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        return abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8))
            

def dist_sim(sim, la, lb):
    wa = Counter(la)
    wb = Counter(lb)
    d1 = {x:1 for x in wa}
    d2 = {x:1 for x in wb}
    return sim.calc(d1, d2)

def weighted_dist_sim(sim, lca, lcb):
    wa = Counter(lca)
    wb = Counter(lcb)
    wa = {x: wweight[x] * wa[x] for x in wa}
    wb = {x: wweight[x] * wb[x] for x in wb}
    return sim.calc(wa, wb)
            
            
def generate_features(input_fname, score_fname=None,
                      outf=sys.stdout, out_format="libsvm", with_lsa=True):
    if score_fname:
        scores = [float(x) for x in open(score_fname)]

    for idx, (sa, sb) in enumerate(load_data(input_fname)):
        ##if idx > 5: break
        
        y = 0. if scores is None else scores[idx]
        
        feats = calc_features(sa, sb, with_lsa)
        
        if out_format == "libsvm":
            feats = ' '.join('%d:%f' % (i + 1, x) for i, x in
                             enumerate(feats))
            outf.write("{} {}\n".format(y, feats))
        elif out_format == "tab":
            # tab-separated
            feats = '\t'.join(str(f) for f in feats)
            outf.write("{}\t{}\n".format(y, feats))
        elif out_format == "numpy":
            try:
                X = np.vstack([X, np.array(feats)])
            except NameError:
                X = np.array(feats)
        else:
            raise ValueError("unknown ouptut format: " + repr(out_format))
    
    if out_format == "numpy":
        y = np.array(scores)
        np.savez(outf, X=X, y=y)
                


if __name__ == "__main__":

    # load word counts for IC weighting
    wweight = load_wweight_table("../wordfreq/_wordfreq-STS2012.txt")
    minwweight = min(wweight.values())
    
    with_lsa = False    
    
    # load vector spaces    
    if with_lsa:
        nyt_sim = Sim('_vsm_data/nyt_words.txt', '_vsm_data/nyt_word_vectors.txt')
        wiki_sim = Sim('_vsm_data/wikipedia_words.txt', '_vsm_data/wikipedia_word_vectors.txt')
    
    # create training instances
    train_dir = "../../data/STS2012-train"
    
    for data in "MSRpar", "MSRvid", "SMTeuroparl":
        out_fname = "_npz_data/_STS2012.train.{}.npz".format(data)
        sys.stderr.write("creating {}\n".format(out_fname))
        generate_features("{}/STS.input.{}.txt".format(train_dir, data),
                          "{}/STS.gs.{}.txt".format(train_dir, data),
                          outf=out_fname, 
                          out_format="numpy", 
                          with_lsa=with_lsa)
        
    # create test instances
    test_dir =  "../../data/STS2012-test"  
    
    for data in "MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews":
        out_fname = "_npz_data/_STS2012.test.{}.npz".format(data)
        sys.stderr.write("creating {}\n".format(out_fname))
        generate_features("{}/STS.input.{}.txt".format(test_dir, data),
                          "{}/STS.gs.{}.txt".format(test_dir, data),
                          outf=out_fname, 
                          out_format="numpy", 
                          with_lsa=with_lsa)
