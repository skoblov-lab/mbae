import typing as t

from tensorflow.keras import backend as K, regularizers

from mbae.attention.base import KTensor
from mbae.attention.ops import group_attentions


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
        # return self.lambda_ * K.mean(K.reshape(group_norms, [-1, l_q]), axis=1)
        return self.lambda_ * K.mean(K.reshape(group_norms, [-1, l_q]))

    def get_config(self):
        return {
            'r': self.r, 'alpha': self.alpha, 'lambda_': self.lambda_,
            'epsilon': self.epsilon
        }


def std_gaussian_kld(mean, log_std):
    log_var = 2.0 * log_std
    var = K.exp(log_var)
    # return -0.5 * K.sum(1 + log_var - var - K.square(mean), axis=1)
    return -0.5 * (1 + log_var - var - K.square(mean))


def frobenius_norm(x: KTensor, axes: t.List[int] = None, eps=K.epsilon()):
    """
    The Frobenius (L2 matrix) norm.
    :param x:
    :param axes:
    :param eps:
    :return:
    """
    return K.sqrt(K.sum(K.square(x), axis=axes) + eps)


def kl_loss(mean, log_stddev):
    """
    Modified KL-divergence between a multivariate Gaussian with a diagonal
    covariance matrix and a standard multivaraite Gaussian with a diagonal
    covariance matrix. The actual KL divergence in this case is
    -0.5 * sum(1 + 2*log(stddev) - stddev^2 - mean^2)
    """
    return K.mean(
        -0.5 * K.sum(1 + log_stddev - K.exp(log_stddev) - K.square(mean), axis=1),
        axis=-1
    )


if __name__ == '__main__':
    raise RuntimeError
