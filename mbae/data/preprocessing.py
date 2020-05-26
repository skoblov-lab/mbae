import copy
import operator as op
import typing as t
from itertools import chain, count

import numpy as np
from fn import F

T = t.TypeVar('T')


homogenous = F(map) >> set >> len >> F(op.contains, [0, 1])
flatmap = F(map) >> chain.from_iterable
strictmap = F(map) >> list


int_t = np.int32


class SequenceEncoder:
    """
    Create a callable that encodes text as an integer array
    """
    def __init__(self, alphabet: t.Iterable[str]):
        unique = sorted(set(chain.from_iterable(alphabet)))
        self._mapping = dict(
            (val, key) for key, val in enumerate(unique, 1)
        )
        self._oov = len(self._mapping) + 1

    def __call__(self, sequence: str, dtype=int_t) -> np.ndarray:
        encoded = (self._mapping.get(char, self._oov) for char in sequence)
        return np.fromiter(encoded, dtype, len(sequence))

    @property
    def mapping(self) -> t.Mapping[str, int]:
        return copy.deepcopy(self._mapping)

    @property
    def oov(self) -> int:
        return self._oov


def expand_categories(categories: np.ndarray, dtype) -> np.ndarray:
    """
    Expand a numeric representation into a zero-padded array of ones. For
    example, given `categories = np.array([2, 5, 6])` we will produce
    ```
    np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ])
    ```
    :param categories:
    :param dtype:
    :return:
    """
    ncat = categories.shape[0]
    maxcat = categories.max()
    expanded = np.zeros(maxcat*ncat, dtype=dtype).reshape((ncat, maxcat))
    for i, j in enumerate(categories):
        expanded[i, :j] = 1
    return expanded


def maxshape(arrays: t.Sequence[np.ndarray]) -> t.List[int]:
    """
    :param arrays: a nonempty sequence of arrays; the sequence must be
    homogeneous with respect to dimensionality.
    :raises ValueError: if `arrays` sequence is empty; if arrays have different
    dimensionality.
    """
    if not arrays:
        raise ValueError('`arrays` should not be empty')
    if not homogenous(np.ndim, arrays):
        raise ValueError('`arrays` must have homogeneous dimensionality')
    return list(np.array([array.shape for array in arrays]).max(axis=0))


def stack(arrays: t.Sequence[np.ndarray], shape: t.Optional[t.Sequence[int]],
          dtype, filler=0, trim=False) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Stack N-dimensional arrays with variable sizes across dimensions.
    :param arrays: a nonempty sequence of arrays; the sequence must be
    homogeneous with respect to dimensionality.
    :param shape: target shape to broadcast each array to. The shape must
    specify one integer per dimension – the output will thus have shape
    `[len(arrays), *shape]`. If None the function will infer the maximal size
    per dimension from `arrays`. To infer size for individual dimension(s)
    use -1.
    :param dtype: output data type
    :param filler: a value to fill in the empty space.
    :param trim: trim arrays to fit the `shape`.
    :raises ValueError: if `len(shape)` doesn't match the dimensionality of
    arrays in `arrays`; if an array can't be broadcasted to `shape` without
    trimming, while trimming is disabled; + all cases specified in function
    `maxshape`
    :return: stacked arrays, a boolean mask (empty positions are False).
    >>> from random import choice
    >>> maxlen = 100
    >>> ntests = 10000
    >>> lengths = range(10, maxlen+1, 2)
    >>> arrays = [
    ...    np.random.randint(0, 127, size=choice(lengths)).reshape((2, -1))
    ...    for _ in range(ntests)
    ... ]
    >>> stacked, masks = stack(arrays, [-1, maxlen], np.int)
    >>> all((arr.flatten() == s[m].flatten()).all()
    ...     for arr, s, m in zip(arrays, stacked, masks))
    True
    >>> stacked, masks = stack(arrays, [2, -1], np.int)
    >>> all((arr.flatten() == s[m].flatten()).all()
    ...     for arr, s, m in zip(arrays, stacked, masks))
    True
    """
    def slices(limits: t.Tuple[int], array: np.ndarray) -> t.List[slice]:
        stops = [min(limit, size) for limit, size in zip(limits, array.shape)]
        return [slice(0, stop) for stop in stops]

    if not isinstance(arrays, t.Sequence):
        raise ValueError('`arrays` must be a Sequence object')
    ndim = arrays[0].ndim
    if shape is not None and len(shape) != ndim:
        raise ValueError("`shape`'s dimensionality doesn't match that of "
                         "`arrays`")
    if shape is not None and any(s < 1 and s != -1 for s in shape):
        raise ValueError('the only allowed non-positive value in `shape` is -1')
    # infer size across all dimensions
    inferred = np.array(maxshape(arrays))
    # mix inferred and requested sizes where requested
    limits = (inferred if shape is None else
              np.where(np.array(shape) == -1, inferred, shape))
    # make sure everything fits fine
    if not (shape is None or trim or (inferred <= limits).all()):
        raise ValueError("can't broadcast all arrays to `shape` without "
                         "trimming")
    stacked = np.full([len(arrays), *limits], filler, dtype=dtype)
    mask = np.zeros([len(arrays), *limits], dtype=bool)
    for i, arr, slices_ in zip(count(), arrays, map(F(slices, limits), arrays)):
        op.setitem(stacked, [i, *slices_], op.getitem(arr, slices_))
        op.setitem(mask, [i, *slices_], True)
    stacked[~mask] = filler
    return stacked, mask


def maskfalse(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace False-masked items with zeros.
    >>> array = np.arange(10)
    >>> mask = np.random.binomial(1, 0.5, len(array)).astype(bool)
    >>> masked = maskfalse(array, mask)
    >>> (masked[mask] == array[mask]).all()
    True
    >>> (masked[~mask] == 0).all()
    True
    """
    if not np.issubdtype(mask.dtype, np.bool):
        raise ValueError("Masks are supposed to be boolean")
    copy = array.copy()
    copy[~mask] = 0
    return copy


if __name__ == '__main__':
    raise RuntimeError
