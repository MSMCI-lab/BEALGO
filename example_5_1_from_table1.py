# example_5_1_from_table1.py
import math
from typing import Dict, Tuple
import mpmath as mp

# ---------------------------------------
# 0) Precision
# ---------------------------------------
mp.mp.dps = 60

# ---------------------------------------
# 1) Table 1: exact pmf f^{(ell)}(x1, x2) for ell = 2,5,7
# Rows: X1 = 0..4 ; Cols: X2 = 0..4  (as provided)
# ---------------------------------------
TAB_ELL2 = [
    [0.5344, 0.0525, 0.0115, 0.0034, 0.0004],
    [0.1098, 0.0610, 0.0335, 0.0095, 0.0013],
    [0.0317, 0.0486, 0.0321, 0.0119, 0.0021],
    [0.0082, 0.0148, 0.0139, 0.0074, 0.0018],
    [0.0011, 0.0026, 0.0033, 0.0024, 0.0008],
]
TAB_ELL5 = [
    [0.5344, 0.0525, 0.0115, 0.0025, 0.0004],
    [0.1098, 0.0610, 0.0258, 0.0082, 0.0015],
    [0.0317, 0.0404, 0.0294, 0.0144, 0.0037],
    [0.0065, 0.0129, 0.0160, 0.0128, 0.0052],
    [0.0011, 0.0030, 0.0055, 0.0063, 0.0036],
]
TAB_ELL7 = [
    [0.5344, 0.0525, 0.0115, 0.0025, 0.0004],
    [0.1098, 0.0610, 0.0258, 0.0082, 0.0015],
    [0.0317, 0.0404, 0.0294, 0.0144, 0.0037],
    [0.0065, 0.0129, 0.0160, 0.0127, 0.0052],
    [0.0011, 0.0030, 0.0054, 0.0064, 0.0039],
]

def table_to_pmf(table) -> Dict[Tuple[int,int], float]:
    pmf = {}
    for x1, row in enumerate(table):
        for x2, pr in enumerate(row):
            if pr > 0:
                pmf[(x1, x2)] = float(pr)
    # (Optional) tiny renorm for rounding drift
    total = sum(pmf.values())
    if abs(total - 1.0) > 1e-6:
        for k in pmf:
            pmf[k] /= total
    return pmf

PMF_BY_ELL = {
    2: table_to_pmf(TAB_ELL2),
    5: table_to_pmf(TAB_ELL5),
    7: table_to_pmf(TAB_ELL7),
}

# ---------------------------------------
# 2) Lognormal severity LST φ(s)
#    Prefer fast Gauss–Hermite; fallback to direct integral if NumPy missing
# ---------------------------------------
def phi_lognormal(s: float, mu: float, sigma: float) -> float:
    try:
        import numpy as np
        from numpy.polynomial.hermite import hermgauss
        n = 64
        x, w = hermgauss(n)  # nodes/weights for weight exp(-x^2)
        # E[ e^{-s e^{mu + sigma Z}} ] with Z~N(0,1)
        # Using Hermite (probabilists): ∫ exp(-x^2) f(sqrt(2) x) dx / sqrt(pi)
        s = mp.mpf(s)
        val = mp.mpf('0.0')
        rt2 = mp.sqrt(2)
        for xi, wi in zip(x, w):
            z = rt2 * mp.mpf(xi)
            val += wi * mp.e**(-s * mp.e**(mu + sigma*z))
        return float(val / mp.sqrt(mp.pi))
    except Exception:
        # Fallback: plain integral over R with standard normal pdf
        pdf = lambda z: mp.e**(-0.5*z*z)/mp.sqrt(2*mp.pi)
        integrand = lambda z: mp.e**(-s * mp.e**(mu + sigma*z)) * pdf(z)
        return float(mp.quad(integrand, [-mp.inf, mp.inf]))

def mean_lognormal(mu: float, sigma: float) -> float:
    return math.exp(mu + 0.5*sigma*sigma)

# ---------------------------------------
# 3) Model wrapper: L_S(s), F0, E[S]
# ---------------------------------------
class TwoTypeModel:
    def __init__(self, pmf: Dict[Tuple[int,int], float], mu1, s1, mu2, s2):
        self.pmf = pmf
        self.mu1, self.s1 = mu1, s1
        self.mu2, self.s2 = mu2, s2
    def F0(self) -> float:
        return float(self.pmf.get((0,0), 0.0))
    def L_S(self, s: float) -> float:
        phi1 = phi_lognormal(s, self.mu1, self.s1)
        phi2 = phi_lognormal(s, self.mu2, self.s2)
        tot = mp.mpf('0.0')
        for (x1, x2), pr in self.pmf.items():
            tot += pr * (phi1**x1) * (phi2**x2)
        return float(tot)
    def ES(self) -> float:
        EX1 = sum(x1*pr for (x1,x2), pr in self.pmf.items())
        EX2 = sum(x2*pr for (x1,x2), pr in self.pmf.items())
        EY1 = mean_lognormal(self.mu1, self.s1)
        EY2 = mean_lognormal(self.mu2, self.s2)
        return EX1*EY1 + EX2*EY2

# ---------------------------------------
# 4) Build CDF and H via inverse Laplace (De Hoog)
# ---------------------------------------
def build_cdf(model: TwoTypeModel):
    F0 = model.F0()
    def F(y: float) -> float:
        g = lambda s: model.L_S(s) / s
        return float(F0 + mp.invertlaplace(g, y, method='dehoog'))
    return F

def build_H(model: TwoTypeModel):
    F0 = model.F0()
    def H(v: float) -> float:
        g = lambda s: (1 - F0 - model.L_S(s)) / (s*s)
        return float(mp.invertlaplace(g, v, method='dehoog'))
    return H

def find_var(F, alpha: float, lo: float = 0.0, hi: float = 2e7, tol: float = 1e-6, maxit: int = 80):
    while F(hi) < alpha:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(maxit):
        mid = 0.5*(lo+hi)
        if F(mid) >= alpha:
            hi = mid
        else:
            lo = mid
        if hi - lo <= max(1.0, abs(mid))*tol:
            break
    return hi

def tvar(model: TwoTypeModel, alpha: float):
    F = build_cdf(model)
    H = build_H(model)
    v = find_var(F, alpha)
    ES = model.ES()
    TVaR = v + (ES - H(v)) / (1.0 - alpha)
    return v, TVaR, ES, model.F0()

# ---------------------------------------
# 5) Run Example 5.1 with your parameters
#    Type-1: median 30k, sigma=0.7  => mu=ln(30000)
#    Type-2: median 70k, sigma=0.9  => mu=ln(70000)
# ---------------------------------------
mu1, s1 = math.log(30000.0), 0.7
mu2, s2 = math.log(70000.0), 0.9

for ell in (2, 5, 7):
    model = TwoTypeModel(PMF_BY_ELL[ell], mu1, s1, mu2, s2)
    for alpha in (0.95, 0.99):
        var, tvar_val, ES, F0 = tvar(model, alpha)
        print(f"ell={ell}, alpha={alpha:.2f} | "
              f"VaR={var:,.2f}  TVaR={tvar_val:,.2f}  "
              f"E[S]={ES:,.2f}  F0={F0:.6f}")
