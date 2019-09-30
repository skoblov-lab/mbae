import typing as t

import tensorflow as tf
from keras import backend as K, layers, initializers
from fn import F

from sklab.attention import core, util
from sklab.attention.core import KTensor


class TransformerAttention(core.QKVAttention):
    """
    A generalised version of attention block from the "Attention is All You
    Need" paper.
    """

    @property
    def masker(self) -> t.Optional[core.AttentionMasker]:
        return self.attention.masker

    def __init__(self,
                 r: int,
                 d_r: int,
                 attention: core.QKVAttention,
                 ffn_hid: int,
                 ffn_activation: layers.Activation,
                 ffn_as_cnn: bool = False,
                 dropout: float = None,
                 attention_regulariser: t.Callable[[KTensor], KTensor] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # common
        self.r = r
        self.d_r = d_r
        self.d = d_r * self.r
        self.dropout = dropout
        # attention
        self.attention_regulariser = attention_regulariser
        self.attention = core.MultiHeadAttention(
            attention, self.r, self.d_r, self.attention_regulariser
        )
        self.norm_attention = core.LayerNormalisation()
        # ffn
        self.ffn_hid = ffn_hid
        self.ffn_activation = ffn_activation
        self.ffn_as_cnn = ffn_as_cnn
        self.ffn = core.PositionFFN(self.ffn_activation, self.ffn_hid, self.d,
                                    self.dropout, self.ffn_as_cnn)
        self.norm_ffn = core.LayerNormalisation()

    def call(self,
             inputs: t.List[KTensor],
             attention_mask: KTensor = None,
             **kwargs) -> t.List[KTensor]:
        q, k, v = self.unpack_qkv(inputs)
        att_v, att = self.attention([q, k, v], attention_mask=attention_mask)
        # make a residual connection between Q and (A \times V) and normalise
        att_v_resid = (
            F(layers.Add())
            >> self.norm_attention
        )([q, att_v])
        # apply the FFN, add another residual connection and normalise
        ffn_resid = (
            F(self.ffn)
            >> (lambda ffn_out: layers.Add()([att_v_resid, ffn_out]))
            >> self.norm_ffn
        )(att_v_resid)
        return [ffn_resid, att]

    def compute_output_shape(self, input_shape):
        return self.attention.compute_output_shape(input_shape)


# TODO check that layers passed as arguments do not have trainable parameters

class Encoder(layers.Layer):
    def __init__(self,
                 depth: int,
                 attention_constructor: t.Callable[[int], core.QKVAttention],
                 attention_regulariser: t.Optional[t.Callable[[KTensor], KTensor]],
                 **kwargs):
        """
        :param steps:
        :param r:
        :param d_r:
        :param attention: a function of step number (i.e. an integer
        in range(0, depth)) and returning a QKVAttention instance
        :param ffn_hid:
        :param ffn_activation:
        :param ffn_as_cnn:
        :param dropout:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.depth = depth
        self.attention_regulariser = attention_regulariser
        self.steps = list(map(attention_constructor, range(self.depth)))

        def call(self, inputs: KTensor, **kwargs) -> KTensor:
            pass


class Decoder(layers.Layer):
    pass


class CrossAttention(layers.Layer):
    pass


if __name__ == '__main__':
    raise RuntimeError
