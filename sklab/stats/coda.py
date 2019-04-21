import operator as op
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import skbio


def closure(x: np.ndarray) -> np.ndarray:
    """
    Calculate compositional closure for a given array
    :param x:
    :return:
    """
    return x / x.sum()


def phylogenetic_bipartition(tree: skbio.TreeNode) -> pd.DataFrame:
    """
    Calculate a bipartition sign matrix for a rooted binary phylogenetic tree
    """
    tip_ids = {
        tip.name: i for i, tip in enumerate(tree.tips())
    }

    @lru_cache(maxsize=None)
    def partition(node) -> Tuple[List[str], List[str]]:
        if len(node.children) != 2:
            raise ValueError('the tree must be strictly binary')
        left_child, right_child = node.children
        left_tips = (
            [tip_ids[left_child.name]] if left_child.is_tip() else
            op.add(*partition(left_child))
        )
        right_tips = (
            [tip_ids[right_child.name]] if right_child.is_tip() else
            op.add(*partition(right_child))
        )
        return left_tips, right_tips

    node_ids, clades = zip(*[
        (node.id, map(np.array, partition(node)))
        for node in tree.traverse(include_self=True)
        if not node.is_tip()
    ])
    n_nodes = len(clades)
    n_tips = len(tip_ids)
    sign_matrix = np.zeros((n_nodes, n_tips), dtype=int)
    for i, (left, right) in enumerate(clades):
        sign_matrix[i, left] = -1
        sign_matrix[i, right] = 1
    return pd.DataFrame(
        data=sign_matrix,
        index=node_ids,
        columns=[name for name, _ in sorted(tip_ids.items(), key=op.itemgetter(1))]
    )


def ilr_transform(bipartition: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate an ILR transform matrix with uniform part weights for a given
    bipartition matrix
    """
    sign_matrix = bipartition.values
    n_left = (sign_matrix < 0).sum(axis=1)
    n_right = (sign_matrix > 0).sum(axis=1)
    scaling_factor = np.sqrt((n_left * n_right) / (n_left + n_right))
    balances = sign_matrix.astype(float)
    for i in range(len(sign_matrix)):
        balances[i, balances[i] < 0] *= scaling_factor[i] / n_left[i]
        balances[i, balances[i] > 0] *= scaling_factor[i] / n_right[i]
    return pd.DataFrame(balances, bipartition.index, bipartition.keys())


if __name__ == '__main__':
    raise RuntimeError
