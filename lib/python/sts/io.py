"""
support for reading/writing files in STS format
"""

from codecs import open
from os.path import join
import numpy as np
from numpy.lib.recfunctions import merge_arrays 


# default for maximum size of sentences in chars 
_max_sent_size = 1024

_from_utf8 = lambda s: s.decode("utf-8")
_converters = {0:_from_utf8, 1:_from_utf8}


def read_system_input(filename, max_sent_size=_max_sent_size):
    # unfortunately merge_array doesn't work with "object" as data type,
    # so use fixed size unicode string
    s_type = "U{}".format(max_sent_size) 
    return np.loadtxt(filename,
                      dtype=[("s1", s_type),("s2", s_type)],
                      # Default comment char "#" occurs in text.
                      # seems impossible to switch off comments,
                      # so use some wacky control char (bell)
                      comments="\a",
                      delimiter="\t",
                      converters=_converters) 
    
    
def read_system_output(filename, with_confidence=False):
    if with_confidence:
        return np.loadtxt(filename, dtype=[("output","f"), ("confidence","f")])
    else:
        return np.loadtxt(filename, usecols=(0,), dtype=[("output","f")])


def read_gold_standard(filename):
    return np.loadtxt(filename, dtype=[("gold","f")])


def read(input_fname, gold_fname, output_fname=None, with_confidence=False,
         max_sent_size=_max_sent_size):
    inp = read_system_input(input_fname, max_sent_size=max_sent_size)
    gold = read_gold_standard(gold_fname)
    if output_fname:
        out = read_system_output(output_fname, with_confidence)
        return merge_arrays((inp, gold, out), flatten=True)
    else:
        return merge_arrays((inp, gold), flatten=True)
        

    