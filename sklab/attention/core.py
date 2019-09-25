import operator as op
import typing as t
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
# manually importing Keras from tf._api due to a bug in PyCharm
#from tensorflow._api.v1.keras import backend as K, layers, models, optimizers, \
#    activations, initializers
from keras import backend as K, layers
from fn import F

from attention import util

AttentionBlock = t.Callable[
    [util.KTensor, util.KTensor, util.KTensor],
    t.Tuple[util.KTensor, util.KTensor]
]


class LayerNormalisation(layers.Layer):

    def __init__(self, eps=K.epsilon(), **kwargs):
        self.eps = eps
        self.gamma = None  # set in LaterNormalisation.__build__
        self.beta = None  # set in LaterNormalisation.__build__
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=input_shape[-1:],
            initializer=initializers.Ones(), trainable=True
        )
        self.beta = self.add_weight(
            name='beta', shape=input_shape[-1:],
            initializer=initializers.Zeros(), trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: a Keras tensor
        :param kwargs:
        :return:
        """
        x = inputs
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention:
    """
    Build a subgraph for scaled dot product attention.
    """

    def __init__(self, dropout: float, dtype=K.floatx()):
        # TODO check dropout
        self.dropout = layers.Dropout(dropout) if dropout else None
        self.dtype = dtype

    def __call__(self, q: util.KTensor, k: util.KTensor, v: util.KTensor) \
            -> t.Tuple[util.KTensor, util.KTensor]:
        r"""
        Argument shape legend: b - batch, l - length (number of entries in a
        sequence), d – entry length (embedding dimensions)
        Given:
            $ Q \in {R}^{ {l}_{q} \times d } $
            $ K \in {R}^{ {l}_{k} \times d } $
        the scale dot-product attention matrix is defined as
        $$
        A = softmax( \frac{ Q \times {K}^{T}) }{ \sqrt{d} } )
        $$
        The block calculates and returns both the attention matrix and the
        $ A \times V $ product
        :param q: a query tensor of shape [b, l_q,  d]
        :param k: a key tensor of shape [b, l_k, d]
        :param v: a value tensor of shape [b, l_v, d], such that l_v == l_k
        :return: the $ A \\times V $ tensor of shape [b, l_v, d], a
        """
        d = tf.shape(q)[-1]
        scale = tf.sqrt(tf.cast(d, dtype=self.dtype))
        # Q \times {K}^{T} => shape = [b, l_q, l_k]
        similarity = layers.Lambda(
            lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / scale
        )([q, k])
        att = layers.Activation('softmax')(similarity)
        att_drop = self.dropout(att) if self.dropout else att
        # A \times V => shape = [b, l_v, d]
        att_v = layers.Lambda(lambda x: K.batch_dot(x[0], x[1]))([att_drop, v])
        return att_v, att


class MultiHeadAttention:
    """
    Transform a single-headed attention block into a multi-headed attention
    """

    def __init__(self, attention: AttentionBlock, l_q: int, l_k: int, r: int, d_r: int,
                 dtype=K.floatx()):
        # TODO check d and r compatibility
        # TODO check dropout
        self.attention = attention
        self.l_q = l_q
        self.l_k = l_k
        self.r = r
        self.d_r = d_r
        self.d = d_r * r
        self.dtype = dtype

    def split(self, x: tf.Tensor) -> tf.Tensor:
        """
        Using this instead of functools.partial(util.split_heads, self.r)
        to avoid closures. For more details see docs on util.split_heads
        :param x:
        :return:
        """
        return util.split_heads(self.r, x)

    def merge(self, x: tf.Tensor) -> tf.Tensor:
        """
        Using this instead of functools.partial(util.merge_heads, self.r)
        to avoid closures. For more details see docs on util.merge_heads
        :param x:
        :return:
        """
        return util.merge_heads(self.r, x)

    def group(self, x: tf.Tensor) -> tf.Tensor:
        """
        Using this instead of functools.partial(util.group_attentions, self.r)
        to avoid closures. For more details see docs on util.group_attentions
        :param x:
        :return:
        """
        return util.group_attentions(self.r, x)

    def __call__(self, q: util.KTensor, k: util.KTensor, v: util.KTensor) \
            -> t.Tuple[util.KTensor, util.KTensor]:
        """
        :param q:
        :param k:
        :param v:
        :return: returns a grouped attention matrix (for more details see
        util.group_attentions)
        """
        # create linear mappings for Q, K and V
        print('Q', tf.shape(q))
        q_map = layers.Dense(self.d, use_bias=False)
        k_map = layers.Dense(self.d, use_bias=False)
        v_map = layers.Dense(self.d, use_bias=False)
        # transform subspaces and split heads
        q_split = layers.Lambda(self.split)(q_map(q))
        k_split = layers.Lambda(self.split)(k_map(k))
        v_split = layers.Lambda(self.split)(v_map(v))
        print('splits', tf.shape(q_split)[-1], tf.shape(k_split)[-1], tf.shape(v_split)[-1])
        # calculate attention heads
        att_v_split, att_split = self.attention(q_split, k_split, v_split)
        # create a linear mapping for A \times V
        att_v_map = layers.Dense(self.d, use_bias=False)
        # merge heads and apply a linear map
        # att_v_merged = layers.Lambda(self.merge, output_shape=(self.l_k, self.d))(att_v_split)
        att_v_merged = layers.Lambda(
            self.merge,
            output_shape=(lambda s: (_safe_intdiv(s[0], self.r), s[1], self.d))
        )(att_v_split)
        att_v = att_v_map(att_v_merged)
        # att_groups = layers.Lambda(self.group, output_shape=(self.l_q, self.r, self.l_k))(att_split)
        att_groups = layers.Lambda(
            self.group,
            output_shape=(lambda s: (_safe_intdiv(s[0], self.r), s[1], self.r, s[2]))
        )(att_split)
        return att_v, att_groups

def _safe_intdiv(a, b):
    try:
        return a // b
    except TypeError:
        return None

if __name__ == '__main__':
    raise RuntimeError
