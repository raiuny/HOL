from scipy.optimize import fsolve, root_scalar
from scipy.special import lambertw
from typing import Tuple
from math import exp
import numpy as np

# 节点饱和时的服务率 mu_S
def calc_mu_S(p: float, tt: float, tf: float, W: int, K: int) -> float:
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    # alpha = calc_alpha_sym(tt, tf, 20, p)
    ret = 2 * alpha * tt * p * (2 * p - 1) / (2 * p - 1 + W * ( p - 2 ** K * (1 - p) ** (K + 1)))
    return ret

# 节点未饱和时的服务率 mu_U
def calc_mu_U(p: float, tt: float, tf: float, W: int, K: int, lambda1: float) -> float:
    pi_ts = calc_mu_S(p, tt, tf, W, K)
    ret = pi_ts / (1 + (1 - tf / tt * (1 - p) / p) * (pi_ts - lambda1))
    return ret

def calc_alpha_sym(tt, tf, n, p): 
    alpha = 1/(tf + 1 + (tt - tf) * (n * p - n * p ** (n/(n-1))) - tf * p ** (n/(n-1)))
    return alpha


def calc_alpha_base(tt, tf, p):
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    return alpha

def calc_alpha_asym(tt, tf, n1, p1, n2, p2):
    n = n1 + n2
    pp = (p1 ** n1 * p2 ** n2) ** (1/(n-1))
    alpha = 1/(tf + 1 + (tt - tf) * (n1 * p1 + n2 * p2 - n * pp) - tf * pp)
    return alpha


def relu(x):
    return x if x > 0 else 0

def calc_conf(p1, p2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, state_idx):
    # UU: 0   US: 1   SU:2   SS: 3
    mu1 = calc_mu_S(p1, tt, tf, W_1, K_1)
    mu2 = calc_mu_S(p2, tt, tf, W_2, K_2)
    if mu1 < 0 or mu2 < 0:
        return 0
    state_signs = [(1,1), (1,-1), (-1,1), (-1,-1)]
    return 1 - 0.5 * ( relu(state_signs[state_idx][0] * (lambda1 - mu1) ) + relu(state_signs[state_idx][1] * (lambda2 - mu2) ) )

# p
def calc_uu_p_fsovle(n1: int, lambda1: float, n2: int, lambda2: float, tt: float, tf: float)-> Tuple[float, float, bool]:
    """_summary_

    Args:
        n1 (int): number of mlds
        lambda1 (float): mld's lambda
        n2 (int): number of slds
        lambda2 (float): sld's lambda
        tt (float): tau_T
        tf (float): tau_F

    Returns:
        Tuple[float, float, bool]: pL, pS, is_correct
    """
    r = tf / tt
    z = r / (1 - (1 - r) * (n1 * lambda1 + n2 * lambda2))
    A = (1 + 1 / tf) * z
    def x_equation(x):
        return (A * lambda1 * x + 1 - z * lambda1) ** n1 * (A * lambda2 * x + 1 - z * lambda2) ** n2 - x
    ans1 = fsolve(x_equation, 0, maxfev=500)[0]
    ans2 = fsolve(x_equation, 10, maxfev=500)[0]
    err1 = x_equation(ans1)
    err2 = x_equation(ans2)
    p1 = A * lambda1 + (1 - z * lambda1) * 1 / ans1
    if np.abs(err1) > 1e-5:
        return -1, -1, False
    p2 = A * lambda2 + (1 - z * lambda2) * 1 / ans1
    if p1 > 1 or p2 > 1:
        return p1, p2, False
    return p1, p2, True


def calc_uu_p_formula(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float)-> Tuple[float, float, bool]:
    """ 
    Args:
        nmld (int): number of MLDs
        nsld (int): number of SLDs
        mld_lambda: lambda of each link
        tt: tau_T
        tf: tau_F
    Returns:
        pL, pS, uu status or not
    """
    uu = True
    allambda = nmld * mld_lambda + nsld * sld_lambda
    A = (allambda * tf / tt) / (1 - (1 - tf / tt) * allambda)
    B = -(allambda * (1 + tf) / tt) / (1 - (1 - tf / tt) * allambda)
    # pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
    # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
    pL = B / np.real(lambertw(B*exp(-A), 0))
    pS = B / np.real(lambertw(B*exp(-A), -1))
    if not np.isreal(lambertw(B*exp(-A))) or pL >= 1:
        uu = False
    return pL, pS, uu


def calc_ps_p_formula(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    """_summary_

    Args:
        n1 (int): number of mlds
        lambda1 (float): mld's lambda
        n2 (int): number of slds
        lambda2 (float): sld's lambda
        W_1: mld's W
        K_1: mld's K
        W_2: sld's W
        K_2: sld's K
        tt (float): tau_T
        tf (float): tau_F

    Returns:
        p_ps
    """
    # US:
    p_us2 = _calc_ps_p_formula(n2-1, W_2, K_2, n1, lambda1, tt, tf)
    p_us1 = _calc_ps_p_formula(n2, W_2, K_2, n1-1, lambda1, tt, tf)
    # SU:
    p_su1 = _calc_ps_p_formula(n1-1, W_1, K_1, n2, lambda2, tt, tf)
    p_su2 = _calc_ps_p_formula(n1, W_1, K_1, n2-1, lambda2, tt, tf)
    return  p_us1, p_us2, p_su1, p_su2

def calc_ps_p_fsolve(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    # US:
    p_us1 = calc_PS1(n2, n1-1, W_2, K_2, lambda1, tt, tf)
    p_us2 = calc_PS1(n2-1, n1, W_2, K_2, lambda1, tt, tf)
    # SU:
    p_su1 = calc_PS1(n1-1, n2, W_1, K_1, lambda2, tt, tf)
    p_su2 = calc_PS1(n1, n2-1, W_1, K_1, lambda2, tt, tf)
    return p_us1, p_us2, p_su1, p_su2

def calc_PA1(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
    def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) ) ** nMLD *\
                        (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1)))) ** nSLD
    pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
    return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)

def calc_PA2(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
    def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - np.exp(- 2 * nMLD * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) \
                          - 2 * nSLD * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
    pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
    return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)

def calc_ss_p_fsolve(nmld: int, nsld: int, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, bool]:
    """ calculate p in S-S scenario, return p of each link and throughput on each link (both saturated)
    Args:
        M (int): number of links
        nmld (int): number of MLDs
        nsld (List): numbers of SLDs for each link
        kk: kappa of MLDs
        tt: tau_T
        tf: tau_F
    Returns:
        p, is_correct
    """
    ss = True
    pa1, err1 = calc_PA1(nmld, nsld, W_mld, K_mld, W_sld, K_sld)
    pa2, err2 = calc_PA1(nmld, nsld-1, W_mld, K_mld, W_sld, K_sld)
    if np.abs(err1) > 1e-5:
        ss = False
        return -1, -1, False
    return pa1, pa2, ss

def calc_ss_p_formula(nmld: int, nsld: int, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, bool]:
    """
    Returns:
        p_as, is_correct
    """
    ss = True
    A = -4 * (nmld / W_mld + nsld / W_sld)
    B = 2 * (nmld / W_mld + nsld / W_sld)
    p_ss = B / np.real(lambertw(B*exp(-A), 0 ))
    if not np.isreal(lambertw(B*exp(-A))):
        ss = False
    return p_ss , ss
    

# 求解ps方程
def _calc_ps_p_formula(n_s: int, W_s: int, K_s: int, n_u: int, lambda_u: float, tt: float, tf: float) -> float:
    def p_func(p, n_s, W_s, K_s, n_u, lambda_u):
        return p - exp(-(n_u * lambda_u) * (1 + tf - tf * p - (tt - tf) * np.log(p) * p) / (tt * p) - 2 * n_s * (2 * p - 1) / (2 * p - 1 + W_s * (p - 2 ** K_s * (1 - p) ** (K_s + 1))))
    ans = -1
    for p in np.arange(0.9999, 0.0001, -0.0001):
        err = np.abs(p_func(p, n_s, W_s, K_s, n_u, lambda_u))
        if err < 1e-3:
            ans = p
            break
    return ans

def calc_PS1(n_s, n_u, W_s, K_s, lambda_u, tt, tf): # nMLD unsaturated, nSLD saturated
    def pf(p, n_s, n_u, W_s, K_s, lambda_u):
        return p - (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_s * (p - 2 ** K_s * (1 - p) ** (K_s + 1))) ) ** n_s *\
                        (1 - lambda_u * (1 + tf - tf * p - (tt - tf) * np.log(p) * p) / (tt * p)) ** n_u
    ans = -1
    for p in np.arange(0.9999, 0.0001, -0.0001):
        err = np.abs(pf(p, n_s, n_u, W_s, K_s, lambda_u))
        if err < 1e-3:
            ans = p
            break
    return ans


# Delay

def calc_access_delay_u(p: float, tt: float, tf: float, W: int, K: int, ilambda: float)-> Tuple[float, float]:
    """calculate queuing delay, access delay

    Params:
    ilambda: input rate per node per slot 
    tt: tau_T
    tf: tau_F
    W: length of the initial contend window
    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p)) 
    alpha = alpha / (1 - ilambda * (1 + tf / tt * (1 - p) / p))
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) + \
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K+1)]
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (3 * alpha ** 2) for i in range(K+1)]
    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret
    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1-p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1
    
    # according to Geo/G/1's theoretical formula
    ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    queuing_delay = ilambda / tt * (ED0_2 - ED0_1 ) / (2 * (1 - ilambda / tt * ED0_1))
    access_delay = ED0_1
    # ED01_L = tt + (1 + W) / 2
    # print(ED0_1, ED01_L)
    total_delay = queuing_delay + access_delay
    return queuing_delay, access_delay

def calc_access_delay_s(p: float, tt: float, tf: float, W: int, K: int, ilambda: float, mu_S: float = 0)-> Tuple[float, float]:
    """calculate queuing delay, access delay

    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    alpha = alpha / (1 - mu_S * (1 + tf / tt * (1 - p) / p))
    # alpha = 0.086
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) + \
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K+1)]
    # print("G1Y", G1Y)
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (3 * alpha ** 2) for i in range(K+1)]
    # print("G2Y", G2Y)
    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret
    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1-p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1
    # print(G2D01, ED0_1)
    queueing_delay = mu_S / tt * (ED0_2 - ED0_1) / (2 * (1 - mu_S / tt * ED0_1))
    access_delay = ED0_1
    # print("queueing delay", queueing_delay)
    return queueing_delay, access_delay