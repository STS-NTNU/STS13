#!/usr/bin/env python

"""
print total count, removing the breakdown per year

Usage:
./get_total.py googlebooks-eng-all-totalcounts-20120701.txt
"""

import sys

print sum(int(t.split(",")[1])
          for t in open(sys.argv[1]).read().strip().split("\t"))
