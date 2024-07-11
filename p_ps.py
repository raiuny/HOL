import numpy as np
from scipy.optimize import fsolve
from utils import calc_alpha_asym, calc_ps_p_formula


def calc_ps_p_fsolve(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    def psf(p, n_u, n_s, W_s, K_s, lambda_u):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_s, p[0], n_u, p[1])
        p_ans[0] = p[0] - (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** n_s *\
                        (1 - lambda_u / (alpha * tt * p[0])) ** (n_u-1)
        p_ans[1] = p[1] - (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** (n_s-1) *\
                        (1 - lambda_u  / (alpha * tt * p[0])) ** n_u
        return p_ans
    p_us = fsolve(psf, [0.9, 0.9], args = (n1, n2, W_2, K_2, lambda1))
    p_su = fsolve(psf, [0.9, 0.9], args = (n2, n1, W_1, K_1, lambda2))
    return p_us, p_su, np.sqrt(np.sum(np.array(psf(p_us, n2, n1, W_2, K_2, lambda1))**2)), np.sqrt(np.sum(np.array(psf(p_su, n1, n2, W_1, K_1, lambda2))**2))

def calc_ps_p_fsovle_iter(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    def psf(p, n_u, n_s, W_s, K_s, lambda_u):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_s, p[0], n_u, p[1])
        p_ans[0] = (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** n_s *\
                        (1 - lambda_u / (alpha * tt * p[0])) ** (n_u-1)
        p_ans[1] = (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** (n_s-1) *\
                        (1 - lambda_u  / (alpha * tt * p[0])) ** n_u
        err = (p_ans[0] - p[0]) ** 2 + (p_ans[1] - p[1]) ** 2
        return p_ans, err
    # us
    p = [0.9, 0.9]
    for i in range(10000):
        p, err1 = psf(p, n1, n2, W_2, K_2, lambda1)
    p_us = p
    p = [0.9, 0.9]
    for i in range(10000):
        p, err2 = psf(p, n2, n1, W_1, K_1, lambda2)
    p_su = p
    return p_us, p_su, err1, err2
        
        

if __name__ == "__main__":
    n1 = 10
    n2 = 10
    lam1 = 0.1 / 10
    lam2 = 0.8 / 10
    tt = tf = 32
    p_us, p_su, err1, err2 = calc_ps_p_fsovle_iter(n1, lam1, n2, lam2, 16, 6, 16, 6, tt, tf)
    print(p_us, p_su, err1, err2)
    p_us, p_su, err1, err2 = calc_ps_p_fsolve(n1, lam1, n2, lam2, 16, 6, 16, 6, tt, tf)
    print("fsolve: ", p_us, p_su, err1, err2)
    p1, p2, p3, p4 = calc_ps_p_formula(n1, lam1, n2, lam2, 16, 6, 16, 6, tt, tf)
    print(p1, p2, p3, p4)