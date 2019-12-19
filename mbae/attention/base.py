import abc
import typing as t

import tensorflow as tf

A = t.TypeVar('A')
KTensor = t.NewType('KTensor', tf.Tensor)
KTensorShape = t.Tuple[t.Optional[int], ...]


class Block(t.Callable, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._built = False

    @property
    def built(self) -> bool:
        return self._built

    def build(self, inputs, **kwargs):
        self._built = True


if __name__ == '__main__':
    raise RuntimeError
