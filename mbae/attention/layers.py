import typing as t

from keras import layers, backend as K, initializers

from mbae.attention.base import KTensor, KTensorShape
from mbae.attention.ops import split_heads, merge_heads, group_attentions


class LayerNormalisation(layers.Layer):

    def __init__(self, eps=K.epsilon(), **kwargs):
        self.eps = eps
        self.gamma = None  # set in LaterNormalisation.build
        self.beta = None  # set in LaterNormalisation.build
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


class StdGaussianVariationalDense(layers.Layer):
    # TODO
    pass


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


if __name__ == '__main__':
    raise RuntimeError
