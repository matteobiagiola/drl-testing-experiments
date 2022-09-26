from bisect import bisect_left
from typing import List, Tuple

import numpy as np
import scipy.stats as ss


# function to calculate Cohen's d for independent samples
def cohend(a: List[float], b: List[float]) -> Tuple[float, str]:
    a = np.asarray(a)
    b = np.asarray(b)
    # calculate the size of samples
    m, n = len(a), len(b)
    assert m == n, "The two list must be of the same length: {}, {}".format(m, n)
    # calculate the variance of the samples
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((m - 1) * s1 + (n - 1) * s2) / (m + n - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(a), np.mean(b)
    # calculate the effect size
    effect_size = (u1 - u2) / s
    magnitude = "large"
    if abs(effect_size) < 0.50:
        magnitude = "small"
    elif 0.50 <= abs(effect_size) < 0.80:
        magnitude = "medium"

    return effect_size, magnitude


def vargha_delaney(a: List[float], b: List[float]) -> Tuple[float, str]:
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param a: a numeric list
    :param b: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(a)
    n = len(b)

    assert m == n, "The two list must be of the same length: {}, {}".format(m, n)

    r = ss.rankdata(a + b)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude
