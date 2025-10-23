# depth_compare.py
from be_joint_pmf import joint_pmf_by_types
import numpy as np
from math import pi
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss

# -------- shared model (ensure these match table8.py exactly) --------
V = list(range(1,8))
node_types = {1:"CI/CD",2:"FUS",3:"IoT",4:"SAMM",5:"ERP",6:"AD",7:"SCADA"}
DELTA_EDGES = {(1,2):0.10,(2,3):0.10,(3,4):0.30,(4,7):0.10,(5,6):0.20,(6,7):0.15,(1,4):0.15,(4,6):0.60}
alpha1, alpha5 = 0.12, 0.08
DIRECT_LAW = {
    frozenset(): (1-alpha1)*(1-alpha5),
    frozenset({1}): alpha1*(1-alpha5),
    frozenset({5}): (1-alpha1)*alpha5,
    frozenset({1,5}): alpha1*alpha5,
}

# severities (must match table8.py)
SEV = {
    "CI/CD":("lognorm", np.log(30_000.0), 0.6),
    "IoT":  ("lognorm", np.log(25_000.0), 0.6),
    "FUS":  ("lognorm", np.log(40_000.0), 0.7),
    "ERP":  ("lognorm", np.log(80_000.0), 0.9),
    "SCADA":("lognorm", np.log(150_000.0),1.0),
    "SAMM": ("weibull", 0.8, 90_000.0),
    "AD":   ("weibull", 0.9,110_000.0),
}

# quadrature + de Hoog inversion (same as table8.py)
_HX,_HW = hermgauss(64); _LX,_LW = laggauss(64)
def phi_lognorm(s, mu, sigma):
    a = sigma*np.sqrt(2.0); z = mu + a*_HX
    vals = np.exp(-s*np.exp(z))
    return (a/np.sqrt(pi))*np.sum(_HW*vals)
def phi_weibull(s, k, lam):
    g = np.exp(-s*lam*(_LX**(1.0/k)))
    return np.sum(_LW*g)
def severity_phi(typ, s):
    kind,a,b = SEV[typ]
    return phi_lognorm(s,a,b) if kind=="lognorm" else phi_weibull(s,a,b)

def be_counts_and_exact(L):
    pmf_counts, meta = joint_pmf_by_types(
        depth=L, V=V, node_types=node_types, Q=DELTA_EDGES,
        direct_law=DIRECT_LAW, return_exact_sets=True, normalize=True)
    return pmf_counts, meta["type_order"]

def build_LY(pmf_counts, type_order):
    def LY(s):
        phi = [severity_phi(t, s) for t in type_order]
        tot = 0.0+0.0j
        for counts, p in pmf_counts.items():
            prod = 1.0+0.0j
            for i,x in enumerate(counts):
                if x: prod *= (phi[i]**x)
            tot += p*prod
        return tot
    return LY

def dehoog(F, t, gamma=1e-4, N=64):
    if t<=0: return 0.0
    h = pi/t; s0 = gamma
    f0 = 0.5*np.real(F(s0))
    terms = [np.real(F(s0+1j*k*h)) for k in range(1,N+1)]
    alt = [((-1)**k)*terms[k-1] for k in range(1,N+1)]
    s = 0.0; p = 1.0
    for a in alt:
        p *= 0.5; s += p*a
    return float((np.exp(s0*t)/t)*(f0+s))
def F_from_LY(LY, y): return max(0.0, min(1.0, dehoog(lambda s:(1.0-LY(s))/s, y)))
def H_from_LY(LY, y):  return max(0.0, dehoog(lambda s:(1.0-LY(s))/(s*s), y))
def find_var_0p99(LY, lo=1e4, hi=2e6):
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if F_from_LY(LY, mid) >= 0.99: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

def tvar_for_L(L):
    pmf_counts, type_order = be_counts_and_exact(L)
    LY = build_LY(pmf_counts, type_order)
    VaR = find_var_0p99(LY)
    # compute E[Y] from counts*means to match your baseline line
    # (optional, only TVaR matters for the comparison)
    HVaR = H_from_LY(LY, VaR)
    # we need E[Y] only if you want to print it; omit for speed otherwise
    # but to mirror your output, compute via exact-set expectations:
    EY = 0.0  # can be filled from exact sets; not needed here
    TVaR = VaR + (EY - HVaR)/0.01  # if EY=0, this underreports; instead compute EY if you print it
    return VaR, HVaR, TVaR

# For a strict TVaR comparison we don't need EY; compute via same formula as your table8:
def EY_from_counts(L):
    pmf_counts, meta = joint_pmf_by_types(
        depth=L, V=V, node_types=node_types, Q=DELTA_EDGES,
        direct_law=DIRECT_LAW, return_exact_sets=True, normalize=True)
    type_order = meta["type_order"]
    muX = np.zeros(len(type_order))
    for c,p in pmf_counts.items(): muX += p*np.array(c, float)
    # means:
    def sev_mean(typ):
        kind,a,b = SEV[typ]
        return float(np.exp(a+0.5*b*b)) if kind=="lognorm" else float((b)*float(__import__("mpmath").gamma(1+1/a)))
    m1 = np.array([float(np.exp(SEV[t][1]+0.5*(SEV[t][2]**2))) if SEV[t][0]=="lognorm" else float(SEV[t][2]*__import__("mpmath").gamma(1+1/SEV[t][1])) for t in type_order])
    return float(muX @ m1)

if __name__=="__main__":
    # L=5
    pmf5, order5 = be_counts_and_exact(5)
    LY5 = build_LY(pmf5, order5)
    VaR5 = find_var_0p99(LY5)
    H5   = H_from_LY(LY5, VaR5)
    EY5  = EY_from_counts(5)
    TVaR5= VaR5 + (EY5 - H5)/0.01
    print(f"L=5: E[Y]={EY5:,.0f}, VaR={VaR5:,.0f}, H(VaR)={H5:,.0f}, TVaR={TVaR5:,.0f}")

    # L=4
    pmf4, order4 = be_counts_and_exact(4)
    LY4 = build_LY(pmf4, order4)
    VaR4 = find_var_0p99(LY4)
    H4   = H_from_LY(LY4, VaR4)
    EY4  = EY_from_counts(4)
    TVaR4= VaR4 + (EY4 - H4)/0.01
    print(f"L=4: E[Y]={EY4:,.0f}, VaR={VaR4:,.0f}, H(VaR)={H4:,.0f}, TVaR={TVaR4:,.0f}")

    print(f"ΔTVaR (L=5→4) = {TVaR5 - TVaR4:,.0f}")
