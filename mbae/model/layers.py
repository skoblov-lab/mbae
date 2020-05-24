import typing as t
import operator as op

# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers, activations
from tensorflow.keras.utils import get_custom_objects

from mbae.model.base import KTensor, KTensorShape
from mbae.model.ops import split_heads, merge_heads, apply_dropout
from mbae.model.regularisers import std_gaussian_kld


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
    (https://arxiv.org/abs/1706.03762). The layer does not implement masking.
    """

    def __init__(self, r: int, dropout: float, **kwargs):
        """
        :param r: the number of attention heads; this number should be a factor
        of embedding size; when r == 1, the layer has no trainable parameters
        and behaves as a regular single-head scaled dot-product attention stack
        :param dropout: applies dropout to attention weights
        :param kwargs: Keras-specific layer arguments
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
        """
        :param input_shape: either a single shape tuple or a list of two shape
        tuples: one for Queries and the other one for Keys (and Values).
        :return:
        """
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
        """
        :param inputs: either a single Keras tensor or a list of two Keras
        tensors: one for Queries and the other one for Keys (and Values).
        :param kwargs:
        :return:
        """
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


class PositionFFN(layers.Layer):
    """
    Position-wise feed-forward network from "Attention is All You Need"
    (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_hidden: int, activation: t.Union[str, t.Callable],
                 **kwargs):
        """
        :param d_hidden: size of the hidden layer
        :param activation: activation function applied to the hidden layer;
        you can pass it as a string or as a Callable; in case you are using
        a custom activation function, don't forget to add it to
        `keras.utils.get_custom_objects()`.
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = activations.get(activation)
        self.d_hidden = d_hidden
        # placeholders for layer parameters
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        super().__init__(**kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config['d_hidden'] = self.d_hidden
        config['activation'] = activations.serialize(self.activation)
        return config

    def build(self, input_shape: KTensorShape):
        d_input = input_shape[-1]
        self.w1 = self.add_weight(
            name='w1',
            shape=(d_input, self.d_hidden),
            initializer='glorot_uniform',
            trainable=True)
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.d_hidden,),
            initializer='zeros',
            trainable=True)
        self.w2 = self.add_weight(
            name='w2',
            shape=(self.d_hidden, d_input),
            initializer='glorot_uniform',
            trainable=True)
        self.b2 = self.add_weight(
            name='b2',
            shape=(d_input,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        hidden = K.bias_add(
            K.dot(inputs, self.w1),
            self.b1,
            data_format='channels_last'
        )
        return K.bias_add(
            K.dot(self.activation(hidden), self.w2),
            self.b2,
            data_format='channels_last'
        )


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
    Unpack Queries, Keys and Values
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
    'PositionFFN': PositionFFN
})


if __name__ == '__main__':
    raise RuntimeError
