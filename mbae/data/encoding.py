import copy
import typing as t
from itertools import chain

import numpy as np
from Bio.Alphabet import IUPAC


PROTEIN_MAPPING = {
    letter: i for i, letter in enumerate(
        chain(IUPAC.extended_protein.letters, '-')
    )
}


def encode_protein(seq: str, reserve_zero=True) -> np.ndarray:
    """
    Transform a protein sequence into an integer array.
    :param seq: a sequence of extended single-letter IUPAC amino acid codes
    and/or gap symbols ('-').
    :param reserve_zero: if True, the lowest amino acid code is set to 1 instead
    of 0; this is useful, because zeros are often used to encode blank positions
    rather than any actual letters of a sequence
    :return:
    """
    try:
        encoded = np.array([PROTEIN_MAPPING[letter] for letter in seq])
        return encoded + 1 if reserve_zero else encoded
    except KeyError:
        raise ValueError(f'Unsupported letters in protein sequence: {seq}')


if __name__ == '__main__':
    raise RuntimeError
