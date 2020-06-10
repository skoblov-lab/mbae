import math
import typing as t

import numpy as np
# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

from mbae_src.model.base import KTensor


def split_heads(r: int, x: KTensor) -> KTensor:
    r"""
    Split sequential data along the last dimension (entries/embeddings)
    into a set of subspaces and flatten the result:
    Given $r \in \mathbb{N}$ and a tensor $X$ of shape $b \times l \times d$,
    where $b$ – batch size, $l$ – the number of entries in a sequences, $d$ is
    entry length (embedding dimensions) such that $d \bmod r = 0$:
    1. Calculate subspace size $d_r = d \bmod r$;
    2. Add a new dimension along the entry (embedding) dimension such that
    each entry (embedding) vector is replaced by an $r \times ${d}_{r}$
    matrix: the resulting tensor will have shape [b, l, r, d_r]. Permute
    dimensions from $[b, l, r, d_r]$ to $[r, b, l, d_r]$. In other
    words, we end up with $r$ consecutive views (entry/embedding splits) of the
    original batch. Each view retains the structure of the original
    batch.
    4. Flatten the output along the 0-axis. The output will be
    $[r \times b, l, d_r]$
    :param r: the number of heads
    :param x: a tensor of shape
    """
    b, l, d = K.int_shape(x)
    d_r = d // r
    # split each entry of shape d into r splits of shape d_r
    splits = K.reshape(x, [-1, l, r, d_r])
    # permute to [r, b, l, d_r]
    head_batches = K.permute_dimensions(splits, [2, 0, 1, 3])
    # drop the r-dimension: [r, b, l, d_r] -> [r*b, l, d_r]
    return K.reshape(head_batches, [-1, l, d_r])


def merge_heads(r: int, x: KTensor) -> KTensor:
    """
    This is the inverse of `split_heads`. For more details, read
    the docs on `split_heads`.
    :param r: the number of heads
    :param x: a tensort of shape $[r \times b, l, {d}_{r}]$
    :return: a tensor of shape $[b, l, {d}_{r}]$
    """
    rb, l, d_r = K.int_shape(x)
    # split the rb dimension into r and b axes, creating
    # a tensor of shape [r, b, l, d_r]
    head_batches = K.reshape(x, [r, -1, l, d_r])
    # permute the axes to [b, l, r, d_r]
    splits = K.permute_dimensions(head_batches, [1, 2, 0, 3])
    # concatenate d_r slices of each original entry to recreate the
    # original entry/embedding structure, i.e. [b, l, d], where
    # d = r * d_r
    return K.reshape(splits, [-1, l, r*d_r])


def group_attentions(r: int, attention_split: KTensor) -> KTensor:
    """
    Reshape a multi-headed attention matrix of shape
    $[r \times b, {l}_{q}, {l}_{k}]$ into $[b, {l}_{q}, r, {l}_{k}]$.
    In other words, group all attention vectors by Q-entries they belong to.
    The grouping respects the corresponding Q-batch structure with respect
    to sequence and entry ordering, though original entries/embeddings are
    replaced by $[r, {l}_{k}]$ matrices of attention vectors (collected across $r$
    attention heads).
    """
    rb, l_q, l_k = K.int_shape(attention_split)
    # separate heads
    heads = K.reshape(attention_split, [r, -1, l_q, l_k])
    # group attention vectors by Q-sequence entries
    return K.permute_dimensions(heads, [1, 2, 0, 3])


def apply_dropout(p: float, x: KTensor, training=None):
    """
    :param p: dropout probability
    :param x: target tensor
    :param training: switch dropout off when not in training mode
    :return:
    """
    if not (isinstance(p, float) and 0 <= p < 1):
        raise ValueError('dropout probability must be a float in [0, 1)')

    def x_prime():
        return K.dropout(x, p)

    return K.in_train_phase(x_prime, x, training) if p else x


def gelu(x):
    """
    The activation from "Gaussian Error Linear Units (GELUs)"
    (https://arxiv.org/pdf/1606.08415.pdf).
    The implementation was taken from https://github.com/kpot/keras-transformer
    """
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))


def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing positional encodings as described in
    "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
    The implementation was taken from https://github.com/kpot/keras-transformer
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2. "
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


def isotropic_gaussian_kld(mean, log_std):
    r"""
    KL divergence between a multivariate Gaussian with a diagonal covariance
    matrix and a standard isotropic Gaussian.
    KL = - 0.5 \sum{ 1 + \log{ \sigma^2 } - \sigma^2 - \mu^2 }
    """
    log_var = 2.0 * log_std
    var = K.exp(log_var)
    return -0.5 * K.sum(1 + log_var - var - K.square(mean), axis=1)


def frobenius_norm(x: KTensor, axes: t.List[int] = None, eps=K.epsilon()):
    """
    The Frobenius (L2 matrix) norm.
    :param x:
    :param axes:
    :param eps:
    :return:
    """
    return K.sqrt(K.sum(K.square(x), axis=axes) + eps)


get_custom_objects().update({
    'gelu': gelu
})


if __name__ == '__main__':
    raise RuntimeError
