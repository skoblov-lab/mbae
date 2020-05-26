import typing as t

# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers, activations
from tensorflow.keras.utils import get_custom_objects

from mbae.model.base import KTensor, KTensorShape
from mbae.model import ops

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


class MultiHeadAttention(layers.Layer):
    """
    Implementing a general purpose multi-head query-key-value scaled dot-product
    attention stack from "Attention is All You Need"
    (https://arxiv.org/abs/1706.03762). The layer does not implement masking.
    """

    def __init__(self, r: int, dropout: float, **kwargs):
        """
        :param r: the number of attention heads; this number should be a factor
        of model dimensionality (embedding size); when r == 1, the layer has no
        trainable parameters and behaves as a regular single-head scaled
        dot-product attention stack
        :param dropout: applies dropout to attention weights
        :param kwargs: Keras-specific layer arguments
        """
        super().__init__(**kwargs)
        if not (isinstance(r, int) and r > 0):
            raise ValueError('r must be a positive integer')
        self._r = r
        if not (isinstance(dropout, float) and 0 <= dropout < 1):
            raise ValueError('dropout must be a float in [0, 1)')
        self.dropout = dropout
        # placeholders for model weights
        self.q_map = None
        self.k_map = None
        self.v_map = None
        self.output_map = None

    @property
    def r(self) -> int:
        """
        The number of attention heads
        :return:
        """
        return self._r

    @property
    def multihead(self) -> bool:
        return self.r > 1

    def build(self, input_shape: t.Union[KTensorShape, t.List[KTensorShape]]):
        """
        :param input_shape: either a single shape tuple or a list of two shape
        tuples: one for Queries and the other one for Keys (and Values).
        :return:
        """
        if isinstance(input_shape, list) and len(input_shape) != 2:
            raise ValueError('...')
        q, k = input_shape if isinstance(input_shape, list) else [input_shape]*2
        d_q = q[-1]
        d_k = k[-1]
        if d_q != d_k:
            raise ValueError(
                f'Different Query ({d_q}) and Key/Value ({d_k}) model '
                f'dimensionality'
            )
        if self.multihead:
            d = d_q
            if d % self.r:
                raise ValueError(
                    f'Model dimensionality {d} is not divisible by the number '
                    f'of attention heads {self.r}'
                )
            self.q_map = self.add_weight(
                name='q_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True
            )
            self.k_map = self.add_weight(
                name='k_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True
            )
            self.v_map = self.add_weight(
                name='v_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True
            )
            self.output_map = self.add_weight(
                name='output_map', shape=(d, d),
                initializer='glorot_uniform', trainable=True
            )
        return super().build(input_shape)

    def attention(self, q, k, v, training=None) -> KTensor:
        ndim = K.cast(K.shape(q)[-1], dtype=K.floatx())
        product = K.batch_dot(q, k, axes=(2, 2))
        weights = K.softmax(product / K.sqrt(ndim))
        weights_dropout = ops.apply_dropout(self.dropout, weights, training)
        return K.batch_dot(weights_dropout, v)

    def call(self, inputs: t.Union[KTensor, t.List[KTensor]], **kwargs):
        """
        :param inputs: either a single Keras tensor or a list of two Keras
        tensors: one for Queries and the other one for Keys (and Values).
        :param kwargs:
        :return:
        """
        if isinstance(inputs, list) and len(inputs) != 2:
            raise ValueError('...')
        q, k = inputs if isinstance(inputs, list) else [inputs]*2
        v = k
        training = kwargs.get('training')
        if self.multihead:
            # apply linear transformations to Q, K and V and split heads
            q_split = ops.split_heads(self.r, K.dot(q, self.q_map))
            k_split = ops.split_heads(self.r, K.dot(k, self.k_map))
            v_split = ops.split_heads(self.r, K.dot(v, self.v_map))
            # apply attention
            output_split = self.attention(q_split, k_split, v_split, training)
            # merge heads back together and apply a linear transformation
            return K.dot(ops.merge_heads(self.r, output_split), self.output_map)
        return self.attention(q, k, v, training)

    def get_config(self):
        config = super().get_config()
        config['r'] = self.r
        config['dropout'] = self.dropout
        return config


class TargetMultiHeadAttention(MultiHeadAttention):
    """
    A very lazy implementation of multi-head Target attention from
    "Hierarchical Convolutional Attention Networks for Text Classification"
    (https://dx.doi.org/10.18653/v1/w18-3002). It is the same as regular
    multi-head attention trainable, albeit with a trainable Query matrix of
    shape (1, d_model).
    """

    def __init__(self, r: int, dropout: float, **kwargs):
        super().__init__(r, dropout, **kwargs)
        # placeholder for query parameters
        self.query = None

    def build(self, input_shape: KTensorShape):
        d_model = input_shape[-1]
        self.query = self.add_weight(
            name='query', shape=(1, d_model),
            initializer='glorot_uniform', trainable=True
        )
        return super().build(input_shape)

    def call(self, inputs: KTensor, **kwargs):
        # this is a very lazy (and inefficient) way of creating a query of shape
        # (batch_size, 1, d_model) with d_model trainable weights:
        # (1) create a tensor of ones with shape (batch_size, 1, 1)
        # (2) multiply it by the weights matrix of shape (1, d_model)
        # `K.ones_like(inputs)` is here to capture batch-size; doing
        # K.ones(K.shape(inputs)[0])[:, None, None] doesn't work, because
        # this inadvertently creates a Graph tensor, which seems to be
        # incompatible with @tf.function
        ones = K.ones_like(inputs)[:, 0, 0]
        query = K.dot(ones[:, None, None], self.query)
        return super().call([query, inputs])


class PositionWiseFFN(layers.Layer):
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
            trainable=True
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.d_hidden,),
            initializer='zeros',
            trainable=True
        )
        self.w2 = self.add_weight(
            name='w2',
            shape=(self.d_hidden, d_input),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(d_input,),
            initializer='zeros',
            trainable=True
        )
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


class FixedPositionalEncoding(layers.Layer):
    """
    Injects positional encodings described in "Attention is All You Need"
    (https://arxiv.org/abs/1706.03762).
    The implementation was taken from https://github.com/kpot/keras-transformer
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = ops.positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale
        )
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


class TrainablePositionalEncoding(layers.Layer):
    """
    Represents trainable positional encodings mentioned in
    "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # placeholder for layer parameters
        self.word_position_encoding = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.word_position_encoding = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='uniform',
            name='word_position_encoding',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.word_position_encoding


class StdIsotropicGaussian(layers.Layer):
    """
    A variational Dense layer approximating a standard isotropic Gaussian
    """
    def __init__(self,
                 units: int,
                 kernel_initializer: t.Union[str, t.Callable] = 'glorot_uniform',
                 bias_initializer: t.Union[str, t.Callable] = 'zeros',
                 **kwargs):
        """
        :param units: the number of hidden units
        :param kernel_initializer:
        :param bias_initializer:
        :param kwargs:
        """
        # TODO check arguments
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
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
        with K.name_scope('activity_regularizer'):
            kld = K.mean(ops.isotropic_gaussian_kld(mean, log_std), axis=None)
        self.add_loss([kld], inputs=[inputs])
        # return a sample
        return self.sample(mean, log_std)

    @staticmethod
    def sample(mean, log_std) -> KTensor:
        shape = K.shape(mean)
        epsilon = K.random_normal(shape, mean=0.0, stddev=1.0)
        std = K.exp(log_std)
        return mean + std * epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config['units'] = self.units
        config['kernel_initializer'] = initializers.serialize(
            self.kernel_initializer
        )
        config['bias_initializer'] = initializers.serialize(
            self.bias_initializer
        )
        return config

    def compute_output_shape(self, input_shape):
        # noinspection PyRedundantParentheses
        return (*input_shape[:-1], self.units)


get_custom_objects().update({
    'LayerNormalisation': LayerNormalisation,
    'MultiHeadAttention': MultiHeadAttention,
    'TargetMultiHeadAttention': TargetMultiHeadAttention,
    'PositionWiseFFN': PositionWiseFFN,
    'FixedPositionalEncoding': FixedPositionalEncoding,
    'TrainablePositionalEncoding': TrainablePositionalEncoding,
    'StdIsotropicGaussian': StdIsotropicGaussian
})


if __name__ == '__main__':
    raise RuntimeError
