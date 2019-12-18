import typing as t

import tensorflow as tf

A = t.TypeVar('A')
KTensor = t.NewType('KTensor', tf.Tensor)
KTensorShape = t.Tuple[t.Optional[int], ...]


if __name__ == '__main__':
    raise RuntimeError
