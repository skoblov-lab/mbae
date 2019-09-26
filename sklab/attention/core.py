import typing as t

import tensorflow as tf
from keras import backend as K, layers, initializers
from fn import F

from sklab.attention import util

# Keras tensor
KTensor = t.NewType('KTensor', tf.Tensor)

# TODO find a way to specify a list of length 3 as input and a list
# TODO of length 2 as output
QKVAttention = t.Callable[[t.List[KTensor]], t.List[KTensor]]
Activation = t.Callable[[KTensor], KTensor]

# TODO implement as Layer and Model objects


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

    def call(self, inputs, **kwargs) -> KTensor:
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


class BatchDot(layers.Layer):
    """
    A wrapper around keras.backend.batch_dot
    """

    def __init__(self, axes: t.Optional[t.Union[int, t.Tuple[int, int]]],
                 **kwargs):
        super().__init__(**kwargs)
        self.axes = axes

    def call(self, inputs, **kwargs) -> KTensor:
        return layers.Lambda(
            lambda x: K.batch_dot(x[0], x[1], axes=self.axes)
        )(inputs)

    def compute_output_shape(self, input_shape):
        x_shape, y_shape = input_shape
        x_ndim, y_ndim = map(len, input_shape)
        if x_ndim < 2 or y_ndim < 2:
            raise ValueError(
                f'Can not do batch_dot on inputs with rank < 2. Received inputs '
                f'with shapes {x_shape} and {y_shape}.'
            )
        x_batch = x_shape[0]
        y_batch = y_shape[0]
        if not (x_batch is None or y_batch is None) and x_batch != y_batch:
            raise ValueError(
                f'Can not do batch_dot on inputs with different batch sizes. '
                f'Received inputs with shapes {x_shape} and {y_shape}.'
            )
        # resolve different axes cases
        axes = (
            [self.axes, self.axes] if isinstance(self.axes, int) else
            list(self.axes) if self.axes is not None else
            [x_ndim - 1, y_ndim - 1] if y_ndim == 2 else
            [x_ndim - 1, y_ndim - 2]
        )
        # make sure all axes are either None or integers
        # TODO rewrite this message and condition
        if any([isinstance(axis, (list, tuple)) for axis in axes]):
            raise ValueError(
                f'Multiple target dimensions are not supported. '
                f'Expected: None, int, (int, int). Received: {axes}.'
            )
        # resolve negative indices
        axes_noneg = [
            axes[0] if axes[0] >= 0 else axes[0] + x_ndim,
            axes[1] if axes[1] >= 0 else axes[1] + y_ndim
        ]
        # make sure we are not multiplying along the batch axis
        if 0 in axes:
            raise ValueError(
                'Can not perform batch_dot over axis 0. If your inputs are not '
                'batched, add a dummy batch dimension to your inputs using '
                'K.expand_dims(x, 0)'
            )
        # use a dummy Dot layer to calculate output shape
        dot = layers.Dot(axes_noneg)
        return dot.compute_output_shape(input_shape)


class SplitHeads(layers.Layer):
    # TODO add docs and argument checks
    def __init__(self, r: int, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def _split(self, x: tf.Tensor) -> tf.Tensor:
        return util.split_heads(self.r, x)

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        return layers.Lambda(
            self._split, output_shape=self.compute_output_shape
        )(inputs)

    def compute_output_shape(self, input_shape):
        b, l, d = input_shape
        d_r = d // self.r
        rb = None if b is None else b * self.r
        return rb, l, d_r


class MergeHeads(layers.Layer):
    # TODO add docs and argument checks
    def __init__(self, r: int, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def _merge(self, x: tf.Tensor) -> tf.Tensor:
        return util.merge_heads(self.r, x)

    def call(self, inputs, **kwargs):
        return layers.Lambda(
            self._merge, output_shape=self.compute_output_shape
        )(inputs)

    def compute_output_shape(self, input_shape):
        rb, l, d_r = input_shape
        d = self.r * d_r
        b = None if rb is None else rb // self.r
        return b, l, d


class GroupAttentions(layers.Layer):
    # TODO add docs and argument checks
    def __init__(self, r: int, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def _group(self, x: tf.Tensor) -> tf.Tensor:
        return util.group_attentions(self.r, x)

    def call(self, inputs, **kwargs) -> KTensor:
        return layers.Lambda(
            self._group, output_shape=self.compute_output_shape
        )(inputs)

    def compute_output_shape(self, input_shape):
        rb, l_q, l_k = input_shape
        b = None if rb is None else rb // self.r
        return b, l_q, self.r, l_k


class PositionFFN(layers.Layer):
    def __init__(self, activation: layers.Activation, d_hid, d_out,
                 as_cnn=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.d_hid = d_hid
        self.d_out = d_out
        if as_cnn:
            self.hidden = layers.Conv1D(self.d_hid, 1, activation=None)
            self.out = layers.Conv1D(self.d_out, activation=None)
        else:
            self.hidden = layers.Dense(self.d_hid, activation=None)
            self.out = layers.Dense(self.d_out, activation=None)

    def call(self, inputs, **kwargs):
        return (F(self.hidden) >> self.activation >> self.out)(inputs)

    def compute_output_shape(self, input_shape):
        return (
            F(self.hidden.compute_output_shape) >> self.out.compute_output_shape
        )(input_shape)


class ScaledDotProductAttention(layers.Layer):
    """
    Build a subgraph for scaled dot product attention.
    """

    def __init__(self, dropout: float, return_drop=False, **kwargs):
        """
        :param dropout:
        :param return_drop: return attention matrix after dropout
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout) if dropout else None
        self.return_drop = return_drop

    def call(self, inputs: t.List[KTensor], **kwargs) -> t.List[KTensor]:
        q, k, v = inputs
        return self._call(q, k, v)

    # TODO merge call and _call
    def _call(self, q: KTensor, k: KTensor, v: KTensor) -> t.List[KTensor]:
        r"""
        Argument shape legend: b - batch, l - length (number of entries in a
        sequence), d – entry length (embedding dimensions)
        Given:
            $ Q \in {R}^{ {l}_{q} \times d } $
            $ K \in {R}^{ {l}_{k} \times d } $
        the scale dot-product attention matrix is defined as
        $$
        A = softmax( \frac{ Q \times {K}^{T}) }{ \sqrt{d} } )
        $$
        Given a value $ V \in {R}^{ {l}_{v} \times d } $, such that
        ${l}_{v} = {l}_{k}$ this layer calculates returns both the attention
         matrix and the $ A \times V $ product
        :param q: a query tensor of shape [b, l_q,  d]
        :param k: a key tensor of shape [b, l_k, d]
        :param v: a value tensor of shape [b, l_v, d], such that l_v == l_k
        :return: $ A \times V $ tensor of shape [b, l_v, d], attention
        matrix of shape [b, l_q, l_k]
        """
        d = K.shape(q)[-1]
        scaling_factor = K.sqrt(K.cast(d, dtype=K.floatx()))
        # Q \times {K}^{T} => shape = [b, l_q, l_k]
        similarity = BatchDot(axes=(2, 2))([q, k])
        att_scaled = layers.Activation('softmax')(similarity / scaling_factor)
        att_drop = self.dropout(att_scaled) if self.dropout else att_scaled
        # A \times V => shape = [b, l_v, d]
        att_v = BatchDot(axes=None)([att_drop, v])
        return [att_v, att_drop if self.return_drop else att_scaled]

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, v_shape = input_shape
        b_q, l_q, d_q = q_shape
        b_k, l_k, d_k = k_shape
        b_v, l_v, d_v = v_shape
        # TODO check that:
        #     1. b_q == b_k == b_v (if they are not None)
        #     2. d_q == d_k; these must not be None
        #     3. l_k == l_v; these must not be None
        #     4. d_v is not None
        # if not (b_q is None or b_k is None) and b_q != b_k:
        #     raise ValueError(
        #         '...'
        #     )
        # if not (d_q is None or d_k is None) and d_q != d_k:
        #     raise ValueError(
        #         '...'
        #     )
        product_shape = (b_q, l_q, d_q)
        attention_shape = (b_q, l_q, l_k)
        return [product_shape, attention_shape]


class MultiHeadAttention(layers.Layer):
    """
    Transform a single-headed attention block into a multi-headed attention
    """

    def __init__(self, attention: QKVAttention, r: int, d_r: int, **kwargs):
        # TODO check d and r compatibility
        # TODO check dropout
        super().__init__(**kwargs)
        self.attention = attention
        self.r = r
        self.d_r = d_r
        self.d = d_r * r
        # head splitter and merger
        self.splitter = SplitHeads(self.r)
        self.merger = MergeHeads(self.r)
        self.att_grouper = GroupAttentions(self.r)
        # create linear mappings for Q, K and V
        self.q_map = layers.Dense(self.d, use_bias=False)
        self.k_map = layers.Dense(self.d, use_bias=False)
        self.v_map = layers.Dense(self.d, use_bias=False)
        # create a linear mapping for A \times V
        self.att_v_map = layers.Dense(self.d, use_bias=False)

    def call(self, inputs, **kwargs) -> t.List[KTensor]:
        q, k, v = inputs
        return self._call(q, k, v)

    def _call(self, q: KTensor, k: KTensor, v: KTensor) -> t.List[KTensor]:
        """
        :param q:
        :param k:
        :param v:
        :return: returns a grouped attention matrix (for more details see
        util.group_attentions)
        """
        # transform subspaces and split heads
        q_split = self.splitter(self.q_map(q))
        k_split = self.splitter(self.k_map(k))
        v_split = self.splitter(self.v_map(v))
        # calculate attention heads
        att_v_split, att_split = self.attention([q_split, k_split, v_split])
        # merge heads and apply a linear map
        att_v_merged = self.merger(att_v_split)
        att_v = self.att_v_map(att_v_merged)
        att_groups = self.att_grouper(att_split)
        return [att_v, att_groups]

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, v_shape = input_shape
        q_split_shape = self.splitter.compute_output_shape(q_shape)
        k_split_shape = self.splitter.compute_output_shape(k_shape)
        v_split_shape = self.splitter.compute_output_shape(v_shape)
        att_v_split_shape, att_split_shape = self.attention.compute_output_shape(
            [q_split_shape, k_split_shape, v_split_shape]
        )
        att_v_merge_shape = self.merger.compute_output_shape(att_v_split_shape)
        att_v_shape = self.att_v_map.compute_output_shape(att_v_merge_shape)
        att_groups_shape = self.att_grouper.compute_output_shape(att_split_shape)
        return [att_v_shape, att_groups_shape]


if __name__ == '__main__':
    raise RuntimeError
