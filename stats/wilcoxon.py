from typing import List, Tuple

import numpy as np
import scipy.stats as stats


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    assert len(a) == len(b), "The two list must be of the same length: {}, {}".format(len(a), len(b))
    return stats.wilcoxon(a, b)


def mannwhitney_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    assert len(a) == len(b), "The two list must be of the same length: {}, {}".format(len(a), len(b))
    return stats.mannwhitneyu(a, b)


def fisher_exact(contingency_table: np.ndarray) -> Tuple[float, float]:
    return stats.fisher_exact(table=contingency_table)


def summary(a: List[float]) -> Tuple[float, float, float, float]:
    array = np.asarray(a)
    if min(array) == max(array) and min(array) == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return array.mean(), array.std(), min(array), max(array)
