import typing as t

from keras import backend as K, regularizers

from mbae.attention.base import KTensor
from mbae.attention.ops import group_attentions


class AttentionFrobeniusNorm(regularizers.Regularizer):

    def __init__(self, r: int, sparse: bool):
        # TODO docs
        # TODO argument checks
        self.r = r
        self.sparse = sparse

    def __call__(self, x: KTensor) -> KTensor:
        # TODO docs
        return attention_regularizer(self.r, self.sparse, x)

    def get_config(self):
        return {'r': self.r, 'sparse': self.sparse}


def frobenius(x: KTensor, axes: t.List[int], eps=K.epsilon()):
    return K.sqrt(K.sum(K.square(x), axis=axes)) + eps


def attention_regularizer(r: int, sparse: bool, attention_weights: KTensor) \
        -> KTensor:
    """
    For each attention group $A$  in `attention_weights` calculate
    $| A \times {A}^{T} - I |$ if `sparse` or $| A \times {A}^{T} |$ otherwise.
    Here $| |$ denotes the Frobenius norm (the L2 matrix norm).
    """
    rb, l_q, l_k = K.int_shape(attention_weights)
    attention_groups = group_attentions(r, attention_weights)
    # flatten the batch axis to produce a tensor of [r, l_k] attention groups
    groups = K.reshape(attention_groups, [-1, r, l_k])
    # calculate $A \times $
    self_sim = K.batch_dot(groups, groups, axes=[2, 2])
    # subtract an identity matrix if `sparse`
    group_norms = frobenius(
        self_sim - K.eye(r) if sparse else self_sim, axes=[1, 2]
    )
    # restore the batch structure
    return K.reshape(group_norms, [-1, l_q])


# def attention_regulariser(r: int, sparse: bool, attention_groups: KTensor) \
#         -> KTensor:
#     """
#     For each attention group $A$  in `attention_groups` calculate
#     $| A \times {A}^{T} - I |$ if `sparse` or $| A \times {A}^{T} |$ otherwise.
#     Here $| |$ denotes the Frobenius norm (the L2 matrix norm).
#     """
#     shape = K.shape(attention_groups)
#     l_q = shape[1]
#     l_k = shape[3]
#     # flatten the batch axis to produce a tensor of [r, l_k] attention groups
#     groups = K.reshape(attention_groups, [-1, r, l_k])
#     # calculate $A \times $
#     self_sim = K.batch_dot(groups, groups, axes=[2, 2])
#     # subtract an identity matrix if `sparse`
#     group_norms = frobenius(
#         self_sim - K.eye(r) if sparse else self_sim, axes=[1, 2]
#     )
#     # restore the batch structure
#     return K.reshape(group_norms, [-1, l_q])


if __name__ == '__main__':
    raise RuntimeError
