import abc
import typing as t

from keras import layers


A = t.TypeVar('A')
KTensor = t.NewType('KTensor', tf.Tensor)


class AttentionMasker(layers.Layer, metaclass=abc.ABCMeta):

    # def __init__(self, mask: KTensor, **kwargs):
    #     super().__init__(**kwargs)
    #     self.mask = mask

    @abc.abstractmethod
    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        pass


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


if __name__ == '__main__':
    raise RuntimeError
