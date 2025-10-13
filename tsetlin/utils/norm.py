import math

def erf(x):
    # Approximation of the error function (Abramowitz and Stegun, formula 7.1.26)
    # with maximal error ~1.5e-7
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y

def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + erf(z))
