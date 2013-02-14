"""
score system output
"""

from scipy.stats.stats import pearsonr



def correlation(gold_standard, system_out):
    """
    calculate UNWEIGHTED Pearson correlation coeficient
    """
    correlation, p_value = pearsonr(gold_standard, system_out) 
    return correlation