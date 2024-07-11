from utils import calc_alpha_asym, calc_uu_p_formula
from scipy.optimize import fsolve, minimize
import numpy as np
import matplotlib.pyplot as plt
def calc_au_p_fsolve1(n1: int, n2: int, lambda1, lambda2, tt: float, tf: float)-> float:
    def pf(p, n_1, n_2, lambda_1, lambda_2):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_1, p[0], n_2, p[1])
        p_ans[0] = p[0] - (1 - lambda_1 / (alpha * tt * p[0])) ** (n_1 - 1) *\
                       (1 - lambda_2  / (alpha * tt * p[1])) ** n_2
        p_ans[1] = p[1] - (1 - lambda_1 / (alpha * tt * p[0])) ** n_1 *\
                       (1 - lambda_2  / (alpha * tt * p[1])) ** (n_2 - 1)
        return p_ans
    p_au = fsolve(pf, [0.9, 0.9], args = (n1, n2, lambda1, lambda2))
    return p_au, np.sqrt(np.sum(np.array(pf(p_au, n1, n2, lambda1, lambda2))**2))

def calc_au_p_fsolve_iter(n1: int, n2: int, lambda1:float, lambda2:float, tt: float, tf: float)-> float:
    def pf(p, n_1, n_2, lambda_1, lambda_2):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_1, p[0], n_2, p[1])
        p_ans[0] = p[0] - (1 - lambda_1 / (alpha * tt * p[0])) ** (n_1 - 1) *\
                       (1 - lambda_2  / (alpha * tt * p[1])) ** n_2
        p_ans[1] = p[1] - (1 - lambda_1 / (alpha * tt * p[0])) ** (n_1 - 1) *\
                       (1 - lambda_2  / (alpha * tt * p[1])) ** n_2
        err = np.sqrt((p_ans[0] - p[0])**2 + (p_ans[1]-p[1])**2)
        return p_ans, err
    # us
    p = [0.9,0.9]
    for i in range(10000):
        p, err = pf(p, n1, n2, lambda1, lambda2)
    p_au = p
    return p_au, err

if __name__ == "__main__":
    tt = 32
    tf = 27
    n_range = np.arange(10, 110, 10)
    p1_fsolve1 = []
    p2_fsolve1 = []
    p1_fsolve2 = []
    p2_fsolve2 = []
    for n in n_range:
        lam1 = 0.786 / n
        p_au1, err1 = calc_au_p_fsolve1(n/2, n/2, lam1, lam1, tt, tf)
        print(f"#1 {n}", p_au1, err1)
        # p_as2, err2 = calc_au_p_fsolve_iter(n/2, n/2, lam1, lam1, tt, tf)
        # print(f"#2 {n}", p_as2, err2)
        p_au, ss = calc_uu_p_formula(n/2, n/2, lam1, lam1,  tt, tf)
        print(f"#3 {n}", p_au, ss)
        # p_as1, p_as2, ss = calc_ss_p_formula(n/2, n/2, 16, 6, 128, 6)
        # print(f"#4 {n}", p_as1,p_as2, ss)
        p1_fsolve1.append(p_au1[0])
        p2_fsolve1.append(p_au1[1])
        p1_fsolve2.append(p_au[0])
        p2_fsolve2.append(p_au[1])
    plt.figure(1)
    plt.plot(n_range, p1_fsolve1, label="p1 fsolve1")
    plt.plot(n_range, p2_fsolve1, label="p2 fsolve1")
    plt.plot(n_range, p1_fsolve2, label="p1 formula")
    plt.plot(n_range, p2_fsolve2, label="p2 formula")
    plt.legend()
    plt.grid()
    plt.ylabel("p (AU) SYM")
    plt.xlabel("n")
    plt.savefig("p_au_asym.png", dpi=400)
        