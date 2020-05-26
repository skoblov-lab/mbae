import math

import tensorflow as tf

from mbae.model.base import KTensor


# noinspection PyTypeChecker
def binomial_negative_log_likelihood(n: int, k: KTensor, p: KTensor):
    """
    Binomial negative log-likelihood. This thing is supposed to be used as a
    loss function in ordinal regression problems, where n represents the
    number of ordinal levels - 1 (i.e., the value of the highest ordinal level),
    k represents observed (true) levels (starting from 0) and p is predicted by
    the model.
    :param n: technically, the total number of trials; here it is the number of
    ordinal levels - 1
    :param k: technically, the number of successes; here it represents observed
    (true) ordinal levels; the lowest level must be denoted by 0; a float Tensor
    :param p: success probability; a float Tensor
    :return:
    """
    n_minus_k = float(n) - k
    log_likelihood = (
        math.lgamma(n + 1)
        - tf.math.lgamma(k + 1.0)
        - tf.math.lgamma(n_minus_k + 1.0)
        + k * tf.math.log(p)
        + n_minus_k * tf.math.log(1.0 - p)
    )
    return - log_likelihood


if __name__ == '__main__':
    raise RuntimeError
