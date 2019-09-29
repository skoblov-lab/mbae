import typing as t

import numpy as np
import tensorflow as tf
from keras import backend as K


A = t.TypeVar('A')


def identity(x: A) -> A:
    return x


def split_heads(r: int, x: tf.Tensor) -> tf.Tensor:
    r"""
    Split sequential data along the last dimension (entries/embeddings)
    into a set of subspaces and flatten the result:
    Given $r \in \mathbb{N}$ and a tensor $X$ of shape $b \times l \times d$,
    where $b$ – batch size, $l$ – the number of entries in a sequences, $d$ is
    entry length (embedding dimensions) such that $d \bmod r = 0$:
    1. Calculate subspace size ${d}_{r} = d \bmod r$;
    2. Add a new dimension along the entry (embedding) dimension such that
    each entry (embedding) vector is replaced by an $r \times ${d}_{r}$
    matrix: the resulting tensor will have shape [b, l, r, d_r]. Permute
    dimensions from $[b, l, r, {d}_{r}]$ to $[r, b, l, {d}_{r}]$. In other
    words, we end up with $r$ consecutive views" (entry/embedding splits) of the
    original batch. Each "view" retains the structure of the original
    batch.
    4. Flatten the output along the 0-axis. The output will be
    $[r \times b, l, {d}_{r}]$
    :param r: the number of heads
    :param x: a tensor of shape
    """
    shape_x = K.shape(x)
    b, l, d = shape_x[0], shape_x[1], shape_x[2]
    d_r = d // r
    # split each entry of shape d into r splits of shape d_r
    splits = K.reshape(x, [-1, l, r, d_r])
    # permute to [r, b, l, d_r]
    head_batches = K.permute_dimensions(splits, [2, 0, 1, 3])
    # drop the r-dimension, i.e.
    # [r, b, l, d_r] -> [r*b, l, d_r]
    return K.reshape(head_batches, [-1, l, d_r])


def merge_heads(r: int, x: tf.Tensor) -> tf.Tensor:
    """
    This is the inverse of `split_heads`. For more details, read
    the docs on `split_heads`.
    :param r: the number of heads
    :param x: a tensort of shape $[r \times b, l, {d}_{r}]$
    :return: a tensor of shape $[b, l, {d}_{r}]$
    """
    shape_x = K.shape(x)
    rb, l, d_r = shape_x[0], shape_x[1], shape_x[2]
    # split the rb dimension into r and b axes, creating
    # a tensor of shape [r, b, l, d_r]
    head_batches = K.reshape(x, [r, -1, l, d_r])
    # permute the axes to [b, l, r, d_r]
    splits = K.permute_dimensions(head_batches, [1, 2, 0, 3])
    # concatenate d_r slices of each original entry to recreate the
    # original entry/embedding structure, i.e. [b, l, d], where
    # d = r * d_r
    return K.reshape(splits, [-1, l, r*d_r])


def group_attentions(r: int, attention_split: tf.Tensor) -> tf.Tensor:
    """
    Reshape a multi-headed attention matrix of shape
    $[r \times b, {l}_{q}, {l}_{k}]$ into $[b, {l}_{q}, r, {l}_{k}]$.
    In other words, group all attention vectors by Q-entries they belong to.
    The grouping respects the corresponding Q-batch structure with respect
    to sequence and entry ordering, though original entries/embeddings are
    replaced by $[r, {l}_{k}]$ matrices of attention vectors (collected across $r$
    attention heads).
    """
    att_shape = K.shape(attention_split)
    rb, l_q, l_k = att_shape[0], att_shape[1], att_shape[2]
    # separate heads
    heads = K.reshape(attention_split, [r, -1, l_q, l_k])
    # group attention vectors by Q-sequence entries
    return K.permute_dimensions(heads, [1, 2, 0, 3])


def frobenius(x, axes, eps=K.epsilon()):
    return K.sqrt(K.sum(K.square(x), axis=axes)) + eps


def attention_regulariser(sparse: bool, attention_groups: tf.Tensor) -> tf.Tensor:
    """
    For each attention group $A$  in `attention_groups` calculate
    $| A \times {A}^{T} - I |$ if `sparse` or $| A \times {A}^{T} |$ otherwise.
    Here $| |$ denotes the Frobenius norm (the L2 matrix norm).
    """
    shape_att = K.shape(attention_groups)
    b, l_q, r, l_k = shape_att[0], shape_att[1], shape_att[2], shape_att[3]
    # flatten the batch axis to produce a tensor of [r, l_k] attention groups
    groups = K.reshape(attention_groups, [-1, r, l_k])
    # calculate $A \times $
    self_sim = K.batch_dot(groups, groups, axes=[2, 2])
    # subtract an idetity matrix if `sparse`
    group_norms = frobenius(self_sim - K.eye(r) if sparse else self_sim, axes=[1, 2])
    # restore the batch structure
    return K.reshape(group_norms, [b, l_q])



if __name__ == '__main__':
    raise RuntimeError
