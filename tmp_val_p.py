import numpy as np
from scipy.optimize import fsolve, root_scalar
from scipy.special import lambertw
from math import exp
from typing import Tuple
import matplotlib.pyplot as plt
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
    print("lambertw", np.real(lambertw(B*exp(-A), 0)), B*exp(-A),A)
    m = B * exp(-A)
    p3 = A + m - m**2 + 3 /2 * m**3 
    if not np.isreal(lambertw(B*exp(-A))) or pL >= 1:
        uu = False
    return pL, pS, p3, uu

if __name__ == "__main__":
    n1 = 10
    n2 = 10
    lambda1 = 0.01
    lambda2 = 0.01
    tt = 27
    tf = 32
    p1, p2, _ = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
    p3, p4, p5, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
    print(p1, p2)
    print(p3, p4, p5)
    x_range = np.arange(0.001, 0.501, 0.001)
    y_range = []
    z_range = []
    # for x in np.arange(0.001, 0.501, 0.001):
    #     B = -(x * (1 + tf) / tt) / (1 - (1 - tf / tt) * x)
    #     A = (x * tf / tt) / (1 - (1 - tf / tt) * x)
    #     m = B*exp(-A)
    #     y = B / np.real(lambertw(B*exp(-A), 0))
    #     y_range.append(y)
    #     # z_range.append(1+x*tf/tt)
    # plt.plot(x_range, y_range)
    # # plt.plot(x_range, z_range, label="z")
    # plt.legend()
    # plt.grid()
    # plt.show()
    for p in np.arange(0, 0.5, 0.001):
        y = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
        y_range.append(y)
    plt.plot(np.arange(0, 0.5, 0.001), y_range)
    # plt.plot(x_range, z_range, label="z")
    plt.legend()
    plt.grid()
    plt.show()