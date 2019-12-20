import typing as t
import operator as op

from keras import backend as K
from keras.layers import Activation, Conv1D, Dense, Dropout, Lambda
from keras.regularizers import Regularizer
from fn import F

from mbae.attention.base import A, KTensor, KTensorShape, Block
from mbae.attention.layers import LayerNormalisation, ActivityRegularizer, \
    ScaledDotProductAttention, BatchDot, SplitHeads, MergeHeads, GroupAttentions


class DotProductAttention(Block):
    # TODO masking

    def __init__(self,
                 r: int,
                 attention_dropout: float = 0.0,
                 attention_regularizer: Regularizer = None,
                 **kwargs):
        # TODO checks
        # TODO names
        super().__init__(**kwargs)
        self.r = r
        self.attention_dropout = (
            Dropout(attention_dropout) if attention_dropout else identity
        )
        self.attention_regularizer = (
            ActivityRegularizer(attention_regularizer)
            if attention_regularizer and self.multi_head else
            identity
        )
        # ops for multi-head case
        self.splitter = SplitHeads(r) if self.multi_head else identity
        self.merger = MergeHeads(r) if self.multi_head else identity
        self.grouper = GroupAttentions(self.r) if self.multi_head else identity
        # placeholders for subspace transforms for multi-head case
        self.q_map = None
        self.k_map = None
        self.v_map = None
        self.concat_map = None

    @property
    def multi_head(self) -> bool:
        return self.r > 1

    def build(self, inputs: t.Union[KTensor, t.List[KTensor]], **kwargs):
        # TODO docs
        q, k, v = unpack_qkv(inputs)
        shapes = map(K.int_shape, [q, k, v])
        (l_q, d_q), (l_k, d_k), (l_v, d_v) = map(op.itemgetter(1, 2), shapes)
        if d_q != d_k:
            raise ValueError
        if l_k != l_v:
            raise ValueError
        if any(ndim % self.r for ndim in [d_q, d_v]):
            raise ValueError
        self.q_map = (
            Dense(d_q, activation=None, use_bias=False) if self.multi_head else
            identity
        )
        self.k_map = (
            Dense(d_k, activation=None, use_bias=False) if self.multi_head else
            identity
        )
        self.v_map = (
            Dense(d_v, activation=None, use_bias=False) if self.multi_head else
            identity
        )
        self.concat_map = (
            Dense(d_v, activation=None, use_bias=False) if self.multi_head else
            identity
        )
        super().build(inputs, **kwargs)

    def __call__(self, inputs: t.Union[KTensor, t.List[KTensor]], **kwargs) \
            -> t.Tuple[KTensor, KTensor]:
        if not self.built:
            self.build(inputs, **kwargs)
        q, k, v = unpack_qkv(inputs)
        # transform subspaces and split heads
        q_split = self.splitter(self.q_map(q))
        k_split = self.splitter(self.k_map(k))
        v_split = self.splitter(self.v_map(v))
        # calculate attention weights, apply attention activity 
        # regularisation and dropout
        weights = self.attention_regularizer(
            ScaledDotProductAttention()([q_split, k_split])
        )
        weights_dropped = self.attention_dropout(weights)
        # apply weights to V splits, concatenate attention heads and
        # group attentions
        v_split_weighted = BatchDot(axes=None)([weights_dropped, v_split])
        v_concat = self.merger(v_split_weighted)
        return self.concat_map(v_concat), self.grouper(weights_dropped)


class PositionFFN(Block):

    def __init__(self, activation: Activation, d_hid: int,
                 hidden_dropout: float = 0.0, convolutional=False, **kwargs):
        super().__init__(**kwargs)
        # TODO argchecks
        self.activation = activation
        self.d_hid = d_hid
        self.convolutional = convolutional
        if not 0 <= hidden_dropout <= 1:
            raise ValueError(
                f'Dropout can be a float in [0, 1), received: {hidden_dropout}'
            )
        self.hidden_dropout = Dropout(hidden_dropout) if hidden_dropout else identity
        # placeholders for transformation layers
        self.hidden = None
        self.output = None

    def build(self, inputs: KTensor, **kwargs):
        b, l, d = K.int_shape(inputs)
        if self.convolutional:
            self.hidden = Conv1D(self.d_hid, 1, activation=None)
            self.output = Conv1D(d, 1, activation=None)
        self.hidden = Dense(self.d_hid, activation=None)
        self.output = Dense(d, activation=None)
        super().build(inputs, **kwargs)

    def __call__(self, inputs: KTensor, **kwargs) -> KTensor:
        if not self.built:
            self.build(inputs)
        return (
            F(self.hidden)
            >> self.activation
            >> self.hidden_dropout
            >> self.output
        )(inputs)


def unpack_qkv(inputs: t.Union[A, t.List[A]]) -> t.List[A]:
    """
    :param inputs: if `len(inputs) == 1`, then `q = k = v = inputs[0]`;
    if `len(inputs) == 2`, then `q = inputs[0]` and k = v = inputs[1]`;
    if `len(inputs) == 3`, then `q, k, v = inputs`
    :return:
    """
    inputs_ = inputs if isinstance(inputs, list) else [inputs]
    nargs = len(inputs_)
    if not 1 <= nargs <= 3:
        raise ValueError('...')
    q, k, v = (
        inputs_ if nargs == 3 else
        [inputs_[0], inputs_[1], inputs_[1]] if nargs == 2 else
        inputs_ * 3
    )
    return [q, k, v]


def identity(x: A) -> A:
    return x


if __name__ == '__main__':
    raise RuntimeError
