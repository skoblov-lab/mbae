import abc
import typing as t

import tensorflow as tf
from keras import backend as K, layers, initializers
from fn import F

from sklab.attention import util

# Keras tensor
KTensor = t.NewType('KTensor', tf.Tensor)

# TODO find a way to specify a list of length 3 as input and a list
# TODO of length 2 as output

A = t.TypeVar('A')

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
        # resolve different `axes` cases
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

# TODO the Residual layer is not particularly useful without a more functional
#      approach to layer composition

# class Residual(layers.Layer):
    #
    # def __init__(self, layer: t.Callable[[KTensor], KTensor], weighted=False,
    #              **kwargs):
    #     super().__init__(**kwargs)
    #     self.layer = layer
    #     self.weighted = weighted
    #     self.alpha = None
    #
    # def build(self, input_shape):
    #     if self.weighted:
    #         self.alpha = self.add_weight(
    #             name='alpha', shape=(1,),
    #             initializer=initializers.Zeros(), trainable=True
    #         )
    #     super().build(input_shape)
    #
    # def call(self, inputs: KTensor, **kwargs) -> KTensor:
    #     if not isinstance(inputs, tf.Tensor):
    #         raise ValueError(
    #             'Residual layers cannot wrap layers of arity != 1'
    #         )
    #     output: KTensor = self.layer(inputs)
    #     if not isinstance(output, tf.Tensor):
    #         raise ValueError(
    #             'Residual layers cannot wrap layers returning multiple '
#                 'tensors'
    #         )
    #     if K.int_shape(inputs) != K.int_shape(output):
    #         raise ValueError('...')
    #     # scale input if necessary
    #     inputs_scaled = (
    #         inputs if not self.weighted else
    #         K.sigmoid(self.alpha) * inputs
    #     )
    #     return layers.Add()([inputs_scaled, output])


class PositionFFN(layers.Layer):
    def __init__(self, activation: layers.Activation, d_hid, d_out,
                 dropout: float = None, as_cnn=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.d_hid = d_hid
        self.d_out = d_out
        self.dropout = layers.Dropout(dropout) if dropout else util.identity
        if as_cnn:
            self.hidden = layers.Conv1D(self.d_hid, 1, activation=None)
            self.out = layers.Conv1D(self.d_out, 1, activation=None)
        else:
            self.hidden = layers.Dense(self.d_hid, activation=None)
            self.out = layers.Dense(self.d_out, activation=None)

    def call(self, inputs, **kwargs):
        return (
            F(self.hidden) >> self.activation >> self.out >> self.dropout
        )(inputs)

    def compute_output_shape(self, input_shape):
        return (
            F(self.hidden.compute_output_shape) >> self.out.compute_output_shape
        )(input_shape)


class AttentionMasker(layers.Layer, metaclass=abc.ABCMeta):

    # def __init__(self, mask: KTensor, **kwargs):
    #     super().__init__(**kwargs)
    #     self.mask = mask

    @abc.abstractmethod
    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        pass


class QueryKeySimilarityMasker(AttentionMasker):

    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        similarity, mask_binary = inputs
        mask = (-1e+9) * (1.0 - K.cast(mask_binary, K.floatx()))
        return layers.Add()([similarity, mask])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class DummyMasker(AttentionMasker):
    """
    Does no masking
    """

    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        similarity, mask = inputs
        return similarity

    def compute_output_shape(self, input_shape):
        similarity_shape, mask_shape = input_shape
        return similarity_shape


class QKVAttention(layers.Layer, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def masker(self) -> t.Optional[AttentionMasker]:
        pass

    @staticmethod
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
            raise ValueError(
                '...'
            )
        q, k, v = (
            inputs_ if nargs == 3 else
            [inputs_[0], inputs_[1], inputs_[1]] if nargs == 2 else
            inputs_ * 3
        )
        return [q, k, v]

    @abc.abstractmethod
    def call(self, inputs: t.List[KTensor], **kwargs) -> t.List[KTensor]:
        pass


class ScaledDotAttention(QKVAttention):
    """
    Build a subgraph for scaled dot product attention.
    """

    def __init__(self, dropout: float = None, return_drop=False,
                 masker: AttentionMasker = None, **kwargs):
        """
        :param dropout:
        :param return_drop: return attention matrix after dropout
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout) if dropout else util.identity
        self.return_drop = return_drop
        self._masker = masker

    @property
    def masker(self) -> t.Optional[AttentionMasker]:
        return self._masker

    def call(self, inputs: t.List[KTensor], attention_mask: KTensor = None,
             **kwargs) -> t.List[KTensor]:
        """
        :param inputs: if `len(inputs) == 1`, then `q = k = v = inputs[0]`;
        if `len(inputs) == 2`, then `q = inputs[0]` and k = v = inputs[1]`;
        if `len(inputs) == 3`, then `q, k, v = inputs`
        :param kwargs:
        :return:
        """
        if attention_mask is not None and self.masker is None:
            raise ValueError('...')
        q, k, v = self.unpack_qkv(inputs)
        return self._call(q, k, v, mask=attention_mask)

    # TODO merge call and _call
    def _call(self, q: KTensor, k: KTensor, v: KTensor, mask=None) -> t.List[KTensor]:
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
        similarity_masked = (
            similarity if mask is None else self.masker([similarity, mask])
        )
        attention = layers.Activation('softmax')(
            similarity_masked / scaling_factor
        )
        attention_drop = self.dropout(attention)
        # A \times V => shape = [b, l_v, d]
        att_v = BatchDot(axes=None)([attention_drop, v])
        return [att_v, attention_drop if self.return_drop else attention]

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, v_shape = self.unpack_qkv(input_shape)
        b_q, l_q, d_q = q_shape
        b_k, l_k, d_k = k_shape
        b_v, l_v, d_v = v_shape
        # TODO check that:
        #     1. b_q == b_k == b_v (if they are not None)
        #     2. d_q == d_k; these must not be None
        #     3. l_k == l_v; these must not be None
        #     4. d_v is not None
        # TODO move shape validation into a separate method
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


class MultiHeadAttention(QKVAttention):
    """
    Transform a single-headed attention block into a multi-headed attention
    """

    def __init__(self, attention: QKVAttention, r: int, d_r: int, **kwargs):
        # TODO check d and r compatibility
        # TODO ? add another dropout?
        super().__init__(**kwargs)
        self.attention = attention
        self.r = r
        self.d_r = d_r
        self.d = d_r * r
        if r > 1:
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
        elif r == 1:
            # single-headed mode: simply wrap self.attention without doing
            # anything except expanding the -2 axis of the attention matrix
            # for its shape to be congruent with the multi-headed version
            identity = layers.Lambda(
                util.identity, output_shape=util.identity
            )
            self.splitter = identity
            self.merger = identity
            self.att_grouper = layers.Lambda(
                lambda x: K.expand_dims(x, axis=-2),
                output_shape=(lambda s: [s[0], s[1], 1, s[2]])
            )
            self.q_map = identity
            self.k_map = identity
            self.v_map = identity
            self.att_v_map = identity
        else:
            raise ValueError('...')

    @property
    def masker(self) -> t.Optional[AttentionMasker]:
        # TODO (?) wrap self.attention.masker in an adapter-layer?
        return self.attention.masker

    def call(self, inputs: t.List[KTensor], attention_mask: KTensor = None,
             **kwargs) -> t.List[KTensor]:
        if attention_mask is not None and self.masker is None:
            raise ValueError('...')

        q, k, v = self.unpack_qkv(inputs)
        return self._call(q, k, v, mask_split=attention_mask)

    def _call(self, q: KTensor, k: KTensor, v: KTensor, mask_split=None) -> t.List[KTensor]:
        """
        :param q:
        :param k:
        :param v:
        :return: returns a grouped attention matrix (for more details see
        util.group_attentions)
        """
        # repeat mask for each head
        mask_split = (
            None if mask_split is None else
            K.repeat_elements(mask_split, self.r, 0)
        )
        # transform subspaces and split heads
        q_split = self.splitter(self.q_map(q))
        k_split = self.splitter(self.k_map(k))
        v_split = self.splitter(self.v_map(v))
        # calculate attention heads
        att_v_split, att_split = self.attention(
            [q_split, k_split, v_split], attention_mask=mask_split
        )
        # merge heads and apply a linear map
        att_v_merged = self.merger(att_v_split)
        att_v = self.att_v_map(att_v_merged)
        att_groups = self.att_grouper(att_split)
        return [att_v, att_groups]

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, v_shape = self.unpack_qkv(input_shape)
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


class Encoder(layers.Layer):
    pass


if __name__ == '__main__':
    raise RuntimeError
