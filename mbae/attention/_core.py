import typing as t

from fn import F
from keras import backend as K, layers

from mbae.attention import util
from mbae.attention._base import AttentionMasker, QKVAttention, \
    Block, KTensorShape
from mbae.attention.layers import BatchDot, SplitHeads, MergeHeads, \
    GroupAttentions
from mbae.attention.base import KTensor

# TODO find a way to specify a list of length 3 as input and a list
# TODO of length 2 as output

A = t.TypeVar('A')


class PositionFFN(Block):

    def __init__(self, activation: layers.Activation, d_hid: int, d_out: int,
                 dropout: float = None, as_cnn=False):
        self.activation = activation
        self.d_hid = d_hid
        self.d_out = d_out
        if not 0 <= dropout <= 1:
            raise ValueError(
                f'Dropout can be None or a float in [0, 1], received: {dropout}'
            )
        self.dropout = layers.Dropout(dropout) if dropout else layers.Lambda(K.identity)
        if as_cnn:
            self.hidden = layers.Conv1D(self.d_hid, 1, activation=None)
            self.out = layers.Conv1D(self.d_out, 1, activation=None)
        else:
            self.hidden = layers.Dense(self.d_hid, activation=None)
            self.out = layers.Dense(self.d_out, activation=None)

    def call(self, inputs: KTensor, **kwargs) -> KTensor:
        return (
            F(self.hidden) >> self.activation >> self.out >> self.dropout
        )(inputs)

    def compute_output_shape(self, input_shape: KTensorShape) -> KTensorShape:
        return (
            F(self.hidden.compute_output_shape) >> self.out.compute_output_shape
        )(input_shape)


class QueryKeySimilarityMasker(AttentionMasker):

    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        similarity, mask_binary = inputs
        mask = (-1e+9) * (1.0 - K.cast(mask_binary, K.floatx()))
        return layers.Add()([similarity, mask])

    def compute_output_shape(self, input_shape: t.List[KTensorShape]) \
            -> KTensorShape:
        similarity_shape, mask_shape = input_shape
        return similarity_shape


class DummyMasker(AttentionMasker):
    """
    Does no masking
    """

    def call(self, inputs: t.List[KTensor], **kwargs) -> KTensor:
        similarity, mask = inputs
        return similarity

    def compute_output_shape(self, input_shape: t.List[KTensorShape]) \
            -> KTensorShape:
        similarity_shape, mask_shape = input_shape
        return similarity_shape


class ScaledDotAttention(QKVAttention):
    """
    Build a subgraph for scaled dot product attention.
    """

    def __init__(self, dropout: float = None, return_drop=False,
                 masker: AttentionMasker = None, **kwargs):
        """
        :param dropout:
        :param return_drop: return attention matrix after dropout
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout) if dropout else layers.Lambda(K.identity)
        self.return_drop = return_drop
        self._masker = masker

    @property
    def masker(self) -> t.Optional[AttentionMasker]:
        return self._masker

    def call(self, inputs: t.List[KTensor], attention_mask: KTensor = None,
             **kwargs) -> t.List[KTensor]:
        """
        :param inputs: if `len(inputs) == 1`, then `q = k = v = inputs[0]`;
        if `len(inputs) == 2`, then `q = inputs[0]` and k = v = inputs[1]`;
        if `len(inputs) == 3`, then `q, k, v = inputs`
        :return:
        """
        if attention_mask is not None and self.masker is None:
            raise ValueError('...')
        q, k, v = self.unpack_qkv(inputs)
        return self._call(q, k, v, mask=attention_mask)

    # TODO merge call and _call
    def _call(self, q: KTensor, k: KTensor, v: KTensor, mask=None) \
            -> t.List[KTensor]:
        r"""
        Argument shape legend: b - batch, l - length (number of entries in a
        sequence), d – entry length (embedding dimensions)
        Given:
            $ Q \in {R}^{ {l}_{q} \times d } $
            $ K \in {R}^{ {l}_{k} \times d } $
        the scale dot-product attention matrix is defined as
        $$
        A = softmax( \frac{ Q \times {K}^{T}) }{ \sqrt{d} } )
        $$
        Given a value $ V \in {R}^{ {l}_{v} \times d } $, such that
        ${l}_{v} = {l}_{k}$ this layer calculates returns both the attention
         matrix and the $ A \times V $ product
        :param q: a query tensor of shape [b, l_q,  d]
        :param k: a key tensor of shape [b, l_k, d]
        :param v: a value tensor of shape [b, l_v, d], such that l_v == l_k
        :return: $ A \times V $ tensor of shape [b, l_v, d], attention
        matrix of shape [b, l_q, l_k]
        """
        d = K.shape(q)[-1]
        scaling_factor = K.sqrt(K.cast(d, dtype=K.floatx()))
        # Q \times {K}^{T} => shape = [b, l_q, l_k]
        similarity = BatchDot(axes=(2, 2))([q, k])
        similarity_masked = (
            similarity if mask is None else self.masker([similarity, mask])
        )
        attention = layers.Activation('softmax')(
            similarity_masked / scaling_factor
        )
        attention_drop = self.dropout(attention)
        # A \times V => shape = [b, l_v, d]
        att_v = BatchDot(axes=None)([attention_drop, v])
        return [att_v, attention_drop if self.return_drop else attention]

    def compute_output_shape(self, input_shape: t.List[KTensorShape]) \
            -> t.List[KTensorShape]:
        q_shape, k_shape, v_shape = self.unpack_qkv(input_shape)
        b_q, l_q, d_q = q_shape
        b_k, l_k, d_k = k_shape
        b_v, l_v, d_v = v_shape
        # TODO check that:
        #     1. b_q == b_k == b_v (if they are not None)
        #     2. d_q == d_k; these must not be None
        #     3. l_k == l_v; these must not be None
        #     4. d_v is not None
        # TODO move shape validation into a separate method
        # if not (b_q is None or b_k is None) and b_q != b_k:
        #     raise ValueError(
        #         '...'
        #     )
        # if not (d_q is None or d_k is None) and d_q != d_k:
        #     raise ValueError(
        #         '...'
        #     )
        product_shape = (b_q, l_q, d_q)
        attention_shape = (b_q, l_q, l_k)
        return [product_shape, attention_shape]


class MultiHeadAttention(QKVAttention):
    """
    Transform a single-headed attention block into a multi-headed attention
    """

    # a regulariser must be layer

    def __init__(self, attention: QKVAttention, r: int, d_r: int, **kwargs):
        """

        :param attention:
        :param r:
        :param d_r:
        # :param regulariser: attention regulariser; the function must accept a
        # tensor of attention groups (see documentation on util.group_attentions)
        # and return a tensor of loss contributions.
        :param kwargs:
        """
        # TODO check d and r compatibility
        # TODO ? add another dropout?
        super().__init__(**kwargs)
        self.attention = attention
        self.r = r
        self.d_r = d_r
        self.d = d_r * r
        if r > 1:
            # head splitter and merger
            self.splitter = SplitHeads(self.r)
            self.merger = MergeHeads(self.r)
            self.att_grouper = GroupAttentions(self.r)
            # create linear mappings for Q, K and V
            self.q_map = layers.Dense(self.d, use_bias=False)
            self.k_map = layers.Dense(self.d, use_bias=False)
            self.v_map = layers.Dense(self.d, use_bias=False)
            # create a linear mapping for A \times V
            self.att_v_map = layers.Dense(self.d, use_bias=False)
        elif r == 1:
            # single-headed mode: simply wrap self.attention without doing
            # anything except expanding the -2 axis of the attention matrix
            # for its shape to be congruent with the multi-headed version
            identity = layers.Lambda(
                K.identity, output_shape=util.identity
            )
            self.splitter = identity
            self.merger = identity
            self.att_grouper = layers.Lambda(
                lambda x: K.expand_dims(x, axis=-2),
                output_shape=(lambda s: [s[0], s[1], 1, s[2]])
            )
            self.q_map = identity
            self.k_map = identity
            self.v_map = identity
            self.att_v_map = identity
        else:
            raise ValueError('...')

    @property
    def masker(self) -> t.Optional[AttentionMasker]:
        # TODO (?) wrap self.attention.masker in an adapter-layer?
        return self.attention.masker

    def call(self, inputs: t.List[KTensor], attention_mask: KTensor = None,
             **kwargs) -> t.List[KTensor]:
        if attention_mask is not None and self.masker is None:
            raise ValueError('...')

        q, k, v = self.unpack_qkv(inputs)
        return self._call(q, k, v, mask=attention_mask)

    def _call(self, q: KTensor, k: KTensor, v: KTensor, mask=None) \
            -> t.List[KTensor]:
        """
        :param q:
        :param k:
        :param v:
        :return: returns a grouped attention matrix (for more details see
        util.group_attentions)
        """
        # repeat mask for each head
        mask_split = (
            None if mask is None else
            K.repeat_elements(mask, self.r, 0)
        )
        # transform subspaces and split heads
        q_split = self.splitter(self.q_map(q))
        k_split = self.splitter(self.k_map(k))
        v_split = self.splitter(self.v_map(v))
        # calculate attention heads
        att_v_split, att_split = self.attention(
            [q_split, k_split, v_split], attention_mask=mask_split
        )
        # merge heads and apply a linear map
        att_v_merged = self.merger(att_v_split)
        att_v = self.att_v_map(att_v_merged)
        att_groups = self.att_grouper(att_split)
        return [att_v, att_groups]

    def compute_output_shape(self, input_shape: t.List[KTensorShape]) \
            -> t.List[KTensorShape]:
        q_shape, k_shape, v_shape = self.unpack_qkv(input_shape)
        q_split_shape = self.splitter.compute_output_shape(q_shape)
        k_split_shape = self.splitter.compute_output_shape(k_shape)
        v_split_shape = self.splitter.compute_output_shape(v_shape)
        att_v_split_shape, att_split_shape = self.attention.compute_output_shape(
            [q_split_shape, k_split_shape, v_split_shape]
        )
        att_v_merge_shape = self.merger.compute_output_shape(att_v_split_shape)
        att_v_shape = self.att_v_map.compute_output_shape(att_v_merge_shape)
        att_groups_shape = self.att_grouper.compute_output_shape(att_split_shape)
        return [att_v_shape, att_groups_shape]


if __name__ == '__main__':
    raise RuntimeError
