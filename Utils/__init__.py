"""Module providing various utility functions.

Like computing superset covers of attribute set collections,
counting marginals from data, SETrees etc."""

from .gamma import gamma, lngamma
from .Counts import compute_counts_dict, compute_counts_array
from .Counts import dict_to_numpy, convert_counts_dicts2numpyarrays
from .AttrSetCover import AttrSetCover
