import math

from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()


def parametric_power_analysis(effect: float = 0.8, alpha: float = 0.05, power: float = 0.8) -> float:
    return analysis.solve_power(effect, power=power, alpha=alpha)


def parametric_power(effect: float, nobs: int, alpha: float = 0.05) -> float:
    return analysis.power(effect_size=effect, nobs1=nobs, alpha=alpha)


# https://stackoverflow.com/questions/15204070/is-there-a-python-scipy-function-to-determine-parameters-needed-to-obtain-a-ta#
def fisher_power_analysis(p1: float, p2: float, power: float = 0.8, sig: float = 0.05) -> int:
    z = norm.isf([sig / 2])  # two-sided t test
    zp = -1 * norm.isf([power])
    d = p1 - p2
    s = 2 * ((p1 + p2) / 2) * (1 - ((p1 + p2) / 2))
    n = s * ((zp + z) ** 2) / (d ** 2)
    if n[0] != math.inf:
        return int(round(n[0]))
    return n[0]
