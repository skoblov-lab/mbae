import typing as t
import warnings
import operator as op

import tensorflow as tf
from fn import F
from keras import backend as K, layers

from mbae.attention import util


__all__ = ['DotProductAttention']


A = t.TypeVar('A')
KTensor = t.NewType('KTensor', tf.Tensor)
KTensorShape = t.Tuple[t.Optional[int], ...]


class DotProductAttention(layers.Layer):

    def __init__(self, r: int, dropout: float = 0,
                 attention_regularizer: t.Callable[[KTensor], KTensor] = None,
                 **kwargs):
        """
        :param r: the number of attention head; it must be a positive integer.
        :param dropout:
        :param attention_regularizer:
        """
        # TODO docs
        super().__init__(**kwargs)
        if not (isinstance(r, int) and r > 0):
            raise ValueError
        if not (isinstance(dropout, float) and 0 <= dropout < 1):
            raise ValueError
        if r == 1 and attention_regularizer:
            warnings.warn('Regularisation is only active in multi-head mode',
                          RuntimeWarning)
        self.r = r
        self.dropout = dropout
        self.attention_regularizer = attention_regularizer
        # placeholders for subspace transforms used in multi-head attention
        self.q_map: t.Optional[KTensor] = None
        self.k_map: t.Optional[KTensor] = None
        self.v_map: t.Optional[KTensor] = None
        self.concat_map: t.Optional[KTensor] = None

    def build(self, input_shape: t.Union[KTensorShape, t.List[KTensorShape]]):
        shapes = unpack_qkv(input_shape)
        (d_q, l_q), (d_k, l_k), (d_v, l_v) = map(op.itemgetter(1, 2), shapes)
        # validate Q, K, V shape compatibility
        if d_q != d_k:
            raise ValueError
        if l_k != l_v:
            raise ValueError
        # add transformation kernels for multi-head mode
        if self.r > 1:
            self.add_weight(name='q_map', shape=(d_q, d_q),
                            initializer='glorot_uniform', trainable=True)
            self.add_weight(name='k_map', shape=(d_k, d_k),
                            initializer='glorot_uniform', trainable=True)
            self.add_weight(name='v_map', shape=(d_v, d_v),
                            initializer='glorot_uniform', trainable=True)
            self.add_weight(name='concat_map', shape=(d_v, d_v),
                            initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs: t.Union[KTensor, t.List[KTensor]],
             training: t.Optional[bool] = None, **kwargs) -> t.List[KTensor]:
        """
        :param inputs
        :param training: defaults to True (although it is set to None for
        consistency with Keras Layer API)
        """
        # TODO expand docs
        # TODO describe math
        # TODO add attention mask
        q, k, v = unpack_qkv(inputs)
        # calculate attentions
        training_ = training or training is None
        att_v, attention = (
            self._single_head(q, k, v, training=training_) if self.r == 1 else
            self._multi_head(q, k, v, training=training_)
        )
        return [att_v, attention]

    def _single_head(self, q, k, v, training: bool) -> t.Tuple[KTensor, KTensor]:
        d = K.shape(q)[-1]
        scaling_factor = K.sqrt(K.cast(d, dtype=K.floatx()))
        product = K.batch_dot(q, k, axes=(2, 2))
        attention = K.softmax(product / scaling_factor)
        # apply dropout mask if necessary
        attention = self._dropout_mask(attention, training)
        return K.batch_dot(attention, v), attention

    def _multi_head(self, q, k, v, training: bool) -> t.Tuple[KTensor, KTensor]:
        # transform subspaces in Q, K and V
        q_prime = K.dot(q, self.q_map)
        k_prime = K.dot(k, self.k_map)
        v_prime = K.dot(v, self.v_map)
        # split heads and calculate attention
        q_heads, k_heads, v_heads = map(
            F(split_heads, self.r), [q_prime, k_prime, v_prime]
        )
        att_v_heads, att_heads = self._single_head(
            q_heads, k_heads, v_heads, training
        )
        # concatenate heads and transform subspaces
        att_v_concat = merge_heads(self.r, att_v_heads)
        att_v = K.dot(att_v_concat, self.concat_map)
        # group attentions
        att_groups = group_attentions(self.r, att_heads)
        # add regularization term
        self.add_loss(self.attention_regularizer(att_groups), [q, k, v])
        return att_v, att_groups

    def _dropout_mask(self, attention: KTensor, training: bool) -> KTensor:
        if self.dropout:
            dropout_mask = dropout_generator(K.ones_like(attention),
                                             self.dropout, training)
            attention_masked = attention * dropout_mask
            # set learning phase
            attention_masked._uses_learning_phase = training
            return attention_masked
        return attention

    def compute_output_shape(self, input_shape) -> t.List[KTensorShape]:
        # TODO docs
        shape_q, shape_k, shape_v = unpack_qkv(input_shape)
        batch, l_q, d_q = shape_q
        _, l_k, __ = shape_k
        attention_shape = (
            (batch, l_q, l_k) if self.r == 1 else
            (batch, l_q, self.r, d_q)
        )
        return [shape_v, attention_shape]


def dropout_generator(mask: KTensor, prob: float, training: bool) \
        -> t.Callable[[], KTensor]:
    """
    Return a callable that sets a random fraction of values in `mask` to zero
    and rescales it (see keras.backend.dropout).
    :param mask: a mask of ones
    :param prob: dropout probability
    :param training: disable dropout if training == False
    """
    def dropped():
        return K.dropout(mask, prob)

    return K.in_train_phase(dropped, mask, training=training)


def unpack_qkv(inputs: t.Union[t.List[A], A]) -> t.List:
    # TODO update docs to reflect the Union input type
    """
    :param inputs: if `len(inputs) == 1`, then `q = k = v = inputs[0]`;
    if `len(inputs) == 2`, then `q = inputs[0]` and k = v = inputs[1]`;
    if `len(inputs) == 3`, then `q, k, v = inputs`
    :return:
    """
    inputs_ = inputs if isinstance(inputs, list) else [inputs]
    nargs = len(inputs_)
    if not 1 <= nargs <= 3:
        raise ValueError('...')
    q, k, v = (
        inputs_ if nargs == 3 else
        [inputs_[0], inputs_[1], inputs_[1]] if nargs == 2 else
        inputs_ * 3
    )
    return [q, k, v]


def split_heads(r: int, x: KTensor) -> KTensor:
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


def merge_heads(r: int, x: KTensor) -> KTensor:
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
    att_shape = K.shape(attention_split)
    rb, l_q, l_k = att_shape[0], att_shape[1], att_shape[2]
    # separate heads
    heads = K.reshape(attention_split, [r, -1, l_q, l_k])
    # group attention vectors by Q-sequence entries
    return K.permute_dimensions(heads, [1, 2, 0, 3])


def frobenius(x: KTensor, axes: t.List[int], eps=K.epsilon()):
    return K.sqrt(K.sum(K.square(x), axis=axes)) + eps


def attention_sparse_frobenius_norm(attention_groups: KTensor) -> KTensor:
    """
    For each attention group $A$  in `attention_groups` calculate
    $| A \times {A}^{T} - I |$ if `sparse` or $| A \times {A}^{T} |$ otherwise.
    Here $| |$ denotes the Frobenius norm (the L2 matrix norm).
    """
    b, l_q, r, l_k = K.int_shape(attention_groups)
    # flatten the batch axis to produce a tensor of [r, l_k] attention groups
    groups = K.reshape(attention_groups, [-1, r, l_k])
    # calculate $A \times $
    self_sim = K.batch_dot(groups, groups, axes=[2, 2])
    # subtract an identity matrix if `sparse`
    group_norms = frobenius(self_sim - K.eye(r), axes=[1, 2])
    # restore the batch structure
    return K.reshape(group_norms, [-1, l_q])


if __name__ == '__main__':
    raise RuntimeError
