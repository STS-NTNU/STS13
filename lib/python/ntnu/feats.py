"""
ntnu features
"""

dkpro_feats = [
    "WordNGramJaccardMeasure_4_stopword-filtered",
    "WordNGramJaccardMeasure_4", 
    "LongestCommonSubsequenceComparator",
    "CharacterNGramMeasure_4", 
    "CharacterNGramMeasure_2", 
    "CharacterNGramMeasure_3",
    "LongestCommonSubsequenceNormComparator", 
    "WordNGramJaccardMeasure_1",
    "WordNGramJaccardMeasure_2_stopword-filtered", 
    "WordNGramJaccardMeasure_3",
    "WordNGramContainmentMeasure_1_stopword-filtered",
    "LongestCommonSubstringComparator", 
    "GreedyStringTiling_3",
    "WordNGramContainmentMeasure_2_stopword-filtered",
    "ESA_Wikipedia",
    "ESA_Wiktionary",
    "ESA_WordNet",
    "MCS06_Resnik_WordNet",
    "TWSI_MCS06_Resnik_WordNet"
]
    

takelab_feats= [ 
    "tl.number_len", "tl.number_f", "tl.number_subset",
    "tl.case_match_len", "tl.case_match_f",
    "tl.stocks_match_len", "tl.stocks_match_f",
    "tl.n_gram_match_lc_1",
    "tl.n_gram_match_lc_2",
    "tl.n_gram_match_lc_3",
    "tl.n_gram_match_lem_1",
    "tl.n_gram_match_lem_2",
    "tl.n_gram_match_lem_3",
    "tl.wn_sim_lem",
    "tl.weight_word_match_olc",
    "tl.weight_word_match_lem",
    "tl.rel_len_diff_lc",
    "tl.rel_ic_diff_olc"
]

takelab_lsa_feats = [
    "tl.dist_sim_nyt",
    "tl.weight_dist_sim_nyt",
    "tl.weight_dist_sim_wiki"
]    

gleb_feats = [
    #"SemanticWordOrderLeacockAndChodorow",
    "GateSmartMatchSim"
]

lars_feats = [
    #"ParseConfidence",
    #"RelationSimilarityFrameTypeMeasure",
    "RelationSimilarityMeasure",
    #"RelationSimilarityMeasureOneConstituent",
    #"RelationSimilarityNegationMeasure",
    "GraphEditDistance"
    ]

hans_feats = [
    "MultisenseRI_CentroidTermTerm_Measure",
    "MultisenseRI_ContextTermTerm_Measure",
    "MultisenseRI_HASensesTermTerm_Measure",
    "MultisenseRI_MaxSenseTermTerm_Measure",
    "RI_AvgTermTerm_Measure",
    "RI_HungarianAlgorithm_Measure",
    "RI_SentVectors_Norm_Measure",
    "RI_SentVectors_TFIDF_Measure"
    ]

subsem_best_feats = ['subsem-w2v-n4-c2000-w50-min5-raw-cos',
                     'subsem-wiki8-n4-c1024-min20-p16-raw-cos',
                     'subsem-wiki9-n4-c2048-min20-p16-raw-cos',
                     'subsem-wiki9-n5-c1024-min320-p16-raw-cos',
                     'subsem-wiki9-n6-c2048-min1200-p16-raw-cos',
                     'subsem-lsitfidf-wiki8-n4-c2000-min5-raw-cos',
                     'subsem-lsitfidf-wiki9-n4-c1000-min80-raw-cos',
                     'subsem-w2v-n3-c1000-w25-min5-raw-cos']

all_feats = (
    dkpro_feats +
    takelab_feats +
    takelab_lsa_feats +
    gleb_feats +
    hans_feats +
    lars_feats
    )