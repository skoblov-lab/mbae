import abc
import typing as t

import tensorflow as tf
from keras import layers
from keras import backend as K

from mbae.attention import util


A = t.TypeVar('A')
# a Tensorflow Tensor augmented with Keras metadata
KTensor = t.NewType('KTensor', tf.Tensor)
KInputOutput = t.Union[KTensor, t.List[KTensor]]
KTensorShape = t.Tuple[t.Optional[int], ...]
KInputOutputShape = t.Union[KTensorShape, t.List[KTensorShape]]


class Block(metaclass=abc.ABCMeta):

    # @property
    # @abc.abstractmethod
    # def trainable_layers(self) -> t.Dict[str, layers.Layer]:
    #     pass

    @abc.abstractmethod
    def call(self, inputs: KInputOutput, **kwargs) -> KInputOutput:
        pass

    @abc.abstractmethod
    def compute_output_shape(self, input_shape: KInputOutputShape) \
            -> KInputOutputShape:
        pass

    def __call__(self, inputs, **kwargs) -> KInputOutput:
        outputs = self.call(inputs, **kwargs)
        return self.set_output_shape(inputs, outputs)

    def set_output_shape(self, inputs: KInputOutput, outputs: KInputOutput) \
            -> KInputOutput:
        return layers.Lambda(
            (lambda inputs_, outputs_: list(map(K.identity, outputs_)) if isinstance(outputs, list) else K.identity(outputs_)),
            output_shape=self.compute_output_shape,
            arguments={'outputs_': outputs}
        )(inputs)


class AttentionMasker(Block):

    # def __init__(self, mask: KTensor, **kwargs):
    #     super().__init__(**kwargs)
    #     self.mask = mask

    @abc.abstractmethod
    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        pass


class ActivityRegularizer(layers.Layer):

    def __init__(self, activity_regularizer: t.Callable[[KTensor], KTensor],
                 **kwargs):
        super().__init__(**kwargs)
        self.activity_regularizer = activity_regularizer

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        return inputs


class QKVAttention(Block):

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


if __name__ == '__main__':
    raise RuntimeError
