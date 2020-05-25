import typing as t

# noinspection PyPep8Naming
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.utils import get_custom_objects

from mbae.model.base import KTensor
from mbae.model.ops import group_attentions


class AttentionFrobeniusNorm(regularizers.Regularizer):
    """
    Given a batch of multi-head attention weights of shape (r*b, l_q, l_k),
    where b is the batch size, r is the number of attention heads, l_q is the
    query (Q) length and l_k is the key (K) length, group all attention vectors
    corresponding to the same word in Q, and for each attention group $A$ of
    shape (r, l_k), calculate $| A \times {A}^{T} - \alpha \cdot I |$. Here
    $| |$ denotes the Frobenius norm (the L2 matrix norm) and $I$ denotes the
    identity matrix of rank r.
    """

    def __init__(self, r: int, alpha: float, lambda_: float = 1.0,
                 epsilon: float = K.epsilon()):
        """
        :param r: the number of attention heads; a integer > 1
        :param alpha: sparsity level; a float in [0, 1]
        :param lambda_: regularisation strength; a float in (0, 1]
        :param epsilon: the fuzz factor used in numeric expressions
        """
        if not (r > 1 and isinstance(r, int)):
            raise ValueError
        if not (0 < lambda_ <= 1 and isinstance(lambda_, float)):
            raise ValueError
        if not (0 <= alpha <= 1 and isinstance(alpha, float)):
            raise ValueError
        if not isinstance(epsilon, float):
            raise ValueError
        self.r = r
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, x: KTensor) -> KTensor:
        """
        :param x: a Keras tensor of multi-head attention weights. Read the
        documentation on ops.split_heads for more details.
        :return:
        """
        rb, l_q, l_k = K.int_shape(x)
        attention_groups = group_attentions(self.r, x)
        # flatten the batch axis to produce a tensor of [r, l_k] attention
        # groups
        groups = K.reshape(attention_groups, [-1, self.r, l_k])
        # calculate $A \times $
        self_sim = K.batch_dot(groups, groups, axes=[2, 2])
        # subtract an identity matrix if `sparse`
        sparsity_term = self.alpha * K.eye(self.r) if self.alpha else None
        group_norms = frobenius_norm(
            self_sim - sparsity_term if self.alpha else self_sim,
            axes=[1, 2]
        )
        # restore the batch structure
        return self.lambda_ * K.mean(K.reshape(group_norms, [-1, l_q]))

    def get_config(self):
        return {
            'r': self.r, 'alpha': self.alpha, 'lambda_': self.lambda_,
            'epsilon': self.epsilon
        }


def frobenius_norm(x: KTensor, axes: t.List[int] = None, eps=K.epsilon()):
    """
    The Frobenius (L2 matrix) norm.
    :param x:
    :param axes:
    :param eps:
    :return:
    """
    return K.sqrt(K.sum(K.square(x), axis=axes) + eps)


get_custom_objects().update({
    'AttentionFrobeniusNorm': AttentionFrobeniusNorm
})

if __name__ == '__main__':
    raise RuntimeError
