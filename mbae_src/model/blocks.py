import typing as t

from tensorflow.keras import layers

from mbae_src.model.base import KTensor
from mbae_src.model.layers import LayerNormalisation, MultiHeadAttention, \
    PositionWiseFFN


def identity(x):
    return x


class SelfAttentionBlock:

    def __init__(self, r: int, d_ffn: int, activation: t.Union[str, t.Callable],
                 dropout_softmax: float, dropout_att: float, dropout_ffn: float,
                 regularise: float = 0.0):
        """
        :param r: the number of attention heads
        :param d_ffn: hidden layer size in the point-wise FFN stack
        :param activation: activation function applied to the hidden layer
        :param dropout_softmax: dropout applied to softmax attention weights
        :param dropout_att: dropout applied to atetntion output prior to its
        residual connection
        :param dropout_ffn: dropout applied to point-wise FFN output prior to
        its residual connection
        :param regularise: apply attention regularisation; please, refer to
        MultiHeadAttention's documentation for more details
        """
        if not (isinstance(r, int) and r > 0):
            raise ValueError('`r` must be a positive integer')
        if not (isinstance(dropout_att, float) and 0 <= dropout_att < 1):
            raise ValueError('`dropout_att` must be a float in [0, 1)')
        self.attention = MultiHeadAttention(r, dropout_softmax, regularise)
        self.drop_att = layers.Dropout(dropout_att)
        self.norm_att = LayerNormalisation()

        if not (isinstance(d_ffn, int) and d_ffn > 0):
            raise ValueError('`d_ffn` must be a positive integer')
        if not (isinstance(dropout_ffn, float) and 0 <= dropout_ffn < 1):
            raise ValueError('`dropout_ffn` must be a float in [0, 1)')
        self.ffn = PositionWiseFFN(d_ffn, activation)
        self.drop_ffn = layers.Dropout(dropout_ffn) if dropout_ffn else identity
        self.norm_ffn = LayerNormalisation()

    def __call__(self, inputs: KTensor) -> KTensor:
        attention = self.attention(inputs)
        attention_drop = self.drop_att(attention)
        attention_resid = layers.Add()([inputs, attention_drop])
        attention_norm = self.norm_att(attention_resid)
        ffn = self.ffn(attention_norm)
        ffn_drop = self.drop_ffn(ffn)
        ffn_resid = layers.Add()([attention_norm, ffn_drop])
        ffn_norm = self.norm_ffn(ffn_resid)
        return ffn_norm


if __name__ == '__main__':
    raise RuntimeError
