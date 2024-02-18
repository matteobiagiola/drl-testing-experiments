from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()


def parametric_power_analysis(
    effect: float = 0.8, alpha: float = 0.05, power: float = 0.8
) -> float:
    return analysis.solve_power(effect, power=power, alpha=alpha)


def parametric_power(effect: float, nobs: int, alpha: float = 0.05) -> float:
    return analysis.power(effect_size=effect, nobs1=nobs, alpha=alpha)
