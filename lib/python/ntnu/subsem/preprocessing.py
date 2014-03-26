import re

from nltk import ngrams

# Simple preprocessing as used by the published Word2Vec models
# Used for all sublexical semantics models.
#
# Removes all non alphanumeric tokens and transforms numbers to individual words.
def mahoney_clean(text_str):
    text_str = " " + text_str + " "
    text_str = text_str.lower()

    text_str = re.sub('1', ' one ', text_str, flags=re.MULTILINE)
    text_str = re.sub('2', ' two ', text_str, flags=re.MULTILINE)
    text_str = re.sub('3', ' three ', text_str, flags=re.MULTILINE)
    text_str = re.sub('4', ' four ', text_str, flags=re.MULTILINE)
    text_str = re.sub('5', ' five ', text_str, flags=re.MULTILINE)
    text_str = re.sub('6', ' six ', text_str, flags=re.MULTILINE)
    text_str = re.sub('7', ' seven ', text_str, flags=re.MULTILINE)
    text_str = re.sub('8', ' eight ', text_str, flags=re.MULTILINE)
    text_str = re.sub('9', ' nine ', text_str, flags=re.MULTILINE)
    text_str = re.sub('0', ' zero ', text_str, flags=re.MULTILINE)

    text_str = re.sub('[^a-z]', ' ', text_str, flags=re.MULTILINE)
    text_str = re.sub('\s+', ' ', text_str, flags=re.MULTILINE)

    return text_str


# Creates a string with character n-grams from a string.
def sublexicalize(text_str, order=3):
    text_str = re.sub(' ', '_', text_str)
    char_ngrams = ngrams(text_str, order)

    return ' '.join([''.join(token) for token in char_ngrams])


def make_preprocessor(order=None):
    if order:
        return lambda text_str: sublexicalize(mahoney_clean(text_str), order=order)
    else:
        return lambda text_str: mahoney_clean(text_str)
