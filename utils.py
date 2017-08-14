from six.moves import xrange as range

import os
import sys
import numpy as np

last_percent_reported = None

# ALPHABET = "abcdefghijklmnopqrstuvwxyz' "

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def text_to_int_sequence(utterances):
    '''
    :utterances: list((str)) of utterances in batch

    :returns: np.array(np.array(ints))
    '''
    targets = []
    # Readings targets
    for t in utterances:
        target = t.replace(' ', '  ').split(' ')  
        # Adding blank label
        target = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target])
        # Transform char into index
        target = np.array([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                              for x in target])
        targets.append(target)
    return(np.array(targets))


def sparse_tensor_feed(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

