from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


def ttest_ind(a: List[float], b: List[float]) -> Tuple[float, float]:
    assert len(a) == len(b), "The two list must be of the same length: {}, {}".format(
        len(a), len(b)
    )
    return stats.ttest_ind(a=a, b=b)


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    assert len(a) == len(b), "The two list must be of the same length: {}, {}".format(
        len(a), len(b)
    )
    return stats.wilcoxon(a, b)


def mannwhitney_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    # assert len(a) == len(b), "The two list must be of the same length: {}, {}".format(len(a), len(b))
    if np.array_equal(a, b):
        # mannwhitneyu throws a value error when two arrays are equal
        return math.inf, math.inf
    return stats.mannwhitneyu(a, b)


def fisher_exact_2(list_a: List[float], list_b: List[float]) -> Tuple[float, float]:
    num_failures_a = sum(1 if num == 1.0 else 0 for num in list_a)
    num_non_failures_a = len(list_a) - num_failures_a

    num_failures_b = sum(1 if num == 1.0 else 0 for num in list_b)
    num_non_failures_b = len(list_b) - num_failures_b

    contingency_table = np.asarray(
        [[num_failures_a, num_failures_b], [num_non_failures_a, num_non_failures_b]]
    )

    odds_ratio, p_value = stats.fisher_exact(table=contingency_table)
    return odds_ratio, p_value


def fisher_exact(*lists, adjust: bool = True) -> List[Tuple[float, float]]:
    if len(lists) == 2:
        odds_ratio, p_value = fisher_exact_2(list_a=lists[0], list_b=lists[1])
        return [(round(odds_ratio, 2), round(p_value, 4))]
    p_values = []
    odds_ratios = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            odds_ratio, p_value = fisher_exact_2(list_a=lists[i], list_b=lists[j])
            p_values.append(round(p_value, 4))
            odds_ratios.append(round(odds_ratio, 2))

    if adjust:
        (
            reject,
            p_values_corrected,
            corrected_alpha_sidak,
            corrected_alpha_bonferroni,
        ) = multipletests(pvals=p_values)
        return list(
            zip(odds_ratios, [round(p_value, 4) for p_value in p_values_corrected])
        )
    return list(zip(odds_ratios, [round(p_value, 4) for p_value in p_values]))


def summary(a: List[float]) -> Tuple[float, float, float, float]:
    array = np.asarray(a)
    if min(array) == max(array) and min(array) == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return array.mean(), array.std(), min(array), max(array)


def anova_plus_tukey(
    lists: List[List], groups: List[str], alpha: float = 0.05
) -> Optional[Union[TukeyHSDResults]]:
    # test normality and other assumptions of one-way Anova and Tukey first
    f_statistic, p_value = stats.f_oneway(*lists)
    if p_value <= alpha:
        # create DataFrame to hold data
        df = pd.DataFrame(
            {
                "score": [it for lst in lists for it in lst],
                "group": np.repeat(groups, repeats=len(lists[0])),
            }
        )
        return pairwise_tukeyhsd(endog=df["score"], groups=df["group"], alpha=alpha)
    return None
