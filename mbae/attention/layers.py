import typing as t
import operator as op

from tensorflow.keras import layers, backend as K, initializers
from tensorflow.keras.utils import get_custom_objects

from mbae.attention.base import KTensor, KTensorShape
from mbae.attention.ops import split_heads, merge_heads, group_attentions, \
    apply_dropout
from mbae.attention.regularizers import std_gaussian_kld


A = t.TypeVar('A')


class LayerNormalisation(layers.Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    """

    def __init__(self, eps=K.epsilon(), **kwargs):
        self.eps = eps
        # placeholders for layer parameters
        self.gain = None
        self.bias = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(
            name='gain', shape=input_shape[-1:],
            initializer=initializers.Ones(), trainable=True
        )
        self.bias = self.add_weight(
            name='bias', shape=input_shape[-1:],
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
        return self.gain * (x - mean) / (std + self.eps) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['eps'] = self.eps
        return config


class ScaledDotProductAttention(layers.Layer):
    """
    Implementing a general purpose multi-head query-key-value scaled dot-product
    attention stack from "Attention is All You Need"
    (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, r: int, dropout: float, **kwargs):
        """
        :param r: the number of attention heads; this number should be a factor
        of embedding size; when r == 1, the layer has no trainable parameters
        and behaves as a regular single-head scaled dot-product attention stack
        :param dropout: applies dropout to attention weights
        :param kwargs:
        """
        super().__init__(**kwargs)
        if not (isinstance(r, int) and r > 0):
            raise ValueError('r must be a positive integer')
        self.r = r
        if not (isinstance(dropout, float) and 0 <= dropout < 1):
            raise ValueError('dropout must be a float in [0, 1)')
        self.dropout = dropout
        # placeholders for model weights
        self.q_map = None
        self.k_map = None
        self.v_map = None
        self.output_map = None

    @property
    def multihead(self) -> bool:
        return self.r > 1

    def build(self, input_shape: t.Union[KTensorShape, t.List[KTensorShape]]):
        q, k, _ = unpack_qkv(input_shape)
        _, l_q, d_q = q
        _, l_k, d_k = k  # k == v
        if {d_q, d_k} != {d_q}:
            # TODO message
            raise ValueError
        d = d_q
        if d % self.r:
            # TODO message
            raise ValueError
        if self.multihead:
            self.q_map = self.add_weight(
                name='q_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True)
            self.k_map = self.add_weight(
                name='k_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True)
            self.v_map = self.add_weight(
                name='v_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True)
            self.output_map = self.add_weight(
                name='output_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True)
        return super().build(input_shape)

    def attention(self, q, k, v, training=None) -> KTensor:
        # TODO dropout
        ndim = K.cast(K.shape(q)[-1], dtype=K.floatx())
        product = K.batch_dot(q, k, axes=(2, 2))
        weights = K.softmax(product / K.sqrt(ndim))
        weights_dropout = apply_dropout(self.dropout, weights, training)
        return K.batch_dot(weights_dropout, v)

    def call(self, inputs: t.Union[KTensor, t.List[KTensor]], **kwargs):
        # technically V is the same thing as K
        q, k, v = unpack_qkv(inputs)
        training = kwargs.get('training')
        if self.multihead:
            # apply linear transformations to Q, K and V and split heads
            q_split = split_heads(self.r, K.dot(q, self.q_map))
            k_split = split_heads(self.r, K.dot(k, self.k_map))
            v_split = split_heads(self.r, K.dot(v, self.v_map))
            # apply attention
            output_split = self.attention(q_split, k_split, v_split, training)
            # merge heads back together and apply a linear transformation
            return K.dot(merge_heads(self.r, output_split), self.output_map)
        return self.attention(q, k, v, training)

    def get_config(self):
        config = super().get_config()
        config['r'] = self.r
        config['dropout'] = self.dropout
        return config


class BatchDot(layers.Layer):
    """
    A wrapper around keras.backend.batch_dot
    """

    def __init__(self, axes: t.Optional[t.Union[int, t.Tuple[int, int]]],
                 **kwargs):
        super().__init__(**kwargs)
        self.axes = axes

    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        a, b = inputs
        return K.batch_dot(a, b, axes=self.axes)

    def compute_output_shape(self, input_shape: KTensorShape) -> KTensorShape:
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

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        return split_heads(self.r, inputs)

    def compute_output_shape(self, input_shape: KTensorShape) -> KTensorShape:
        b, l, d = input_shape
        d_r = d // self.r
        rb = None if b is None else b * self.r
        return rb, l, d_r


class MergeHeads(layers.Layer):
    # TODO add docs and argument checks
    def __init__(self, r: int, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        return merge_heads(self.r, inputs)

    def compute_output_shape(self, input_shape: KTensorShape) -> KTensorShape:
        rb, l, d_r = input_shape
        d = self.r * d_r
        b = None if rb is None else rb // self.r
        return b, l, d


class GroupAttentions(layers.Layer):
    # TODO add docs and argument checks
    def __init__(self, r: int, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def call(self, inputs, **kwargs) -> KTensor:
        return group_attentions(self.r, inputs)

    def compute_output_shape(self, input_shape: KTensorShape) -> KTensorShape:
        rb, l_q, l_k = input_shape
        b = None if rb is None else rb // self.r
        return b, l_q, self.r, l_k


class StdIsotropicGaussian(layers.Layer):
    # TODO docs
    def __init__(self,
                 units: int,
                 lambda_: float = 1.0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        # TODO check arguments
        super().__init__(**kwargs)
        self.units = units
        self.lambda_ = lambda_
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        # placeholders for weights
        self.mean_kernel = None
        self.mean_bias = None
        self.std_kernel = None
        self.std_bias = None

    def build(self, input_shape: KTensorShape):
        if len(input_shape) < 2:
            raise ValueError
        input_dim = input_shape[-1]
        self.mean_kernel = self.add_weight(shape=(input_dim, self.units),
                                           initializer=self.kernel_initializer,
                                           name='mean_kernel')
        self.mean_bias = self.add_weight(shape=(self.units,),
                                         initializer=self.bias_initializer,
                                         name='mean_bias')
        self.std_kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='std_kernel')
        self.std_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='std_bias')
        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        super().build(input_shape)

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        mean = K.bias_add(K.dot(inputs, self.mean_kernel),
                          self.mean_bias,
                          data_format='channels_last')
        log_std = K.bias_add(K.dot(inputs, self.std_kernel),
                             self.std_bias,
                             data_format='channels_last')
        # add kl divergence as activity regularizer
        if self.lambda_:
            with K.name_scope('activity_regularizer'):
                kld = K.mean(std_gaussian_kld(mean, log_std), axis=None)
            self.add_loss([self.lambda_ * kld], inputs=[inputs])
        # return a sample
        return self.sample(mean, log_std)

    @staticmethod
    def sample(mean, log_std) -> KTensor:
        shape = K.shape(mean)
        epsilon = K.random_normal(shape, mean=0.0, stddev=1.0)
        std = K.exp(log_std)
        return mean + std * epsilon

    def compute_output_shape(self, input_shape):
        # noinspection PyRedundantParentheses
        return (*input_shape[:-1], self.units)


def unpack_qkv(inputs: t.Union[A, t.List[A]]) -> t.List[A]:
    """
    :param inputs: if `len(inputs) == 1`, then `q = k = v = inputs[0]`;
    if `len(inputs) == 2`, then `q = inputs[0]` and k = v = inputs[1]`;
    :return:
    """
    inputs_ = inputs if isinstance(inputs, list) else [inputs]
    nargs = len(inputs_)
    if not 1 <= nargs <= 2:
        raise ValueError('...')
    q, k, v = (
        [inputs_[0], inputs_[1], inputs_[1]] if nargs == 2 else
        inputs_ * 3
    )
    return [q, k, v]


get_custom_objects().update({
    'LayerNormalisation': LayerNormalisation,
    'StdIsotropicGaussian': StdIsotropicGaussian,
    'ScaledDotProductAttention': ScaledDotProductAttention,
    'BatchDot': BatchDot,
    'SplitHeads': SplitHeads,
    'MergeHeads': MergeHeads,
    'GroupAttentions': GroupAttentions
})


if __name__ == '__main__':
    raise RuntimeError
