"""Module providing various utility functions.

Like computing superset covers of attribute set collections,
counting marginals from data, SETrees etc."""

from .Counts import compute_counts_dict, compute_counts_array
from .Counts import dict_to_numpy, convert_counts_dicts2numpyarrays
from .AttrSetCover import AttrSetCover
from .SparseDistr import SparseDistr

__all__ = (compute_counts_dict, compute_counts_array,
           dict_to_numpy, convert_counts_dicts2numpyarrays,
           AttrSetCover, SparseDistr)
