from utils import calc_alpha_asym, calc_ss_p_fsolve_pa
from scipy.optimize import fsolve, minimize
import numpy as np
import matplotlib.pyplot as plt
def calc_as_p_fsolve1(n1: int, n2: int, W_1: int, K_1: int, W_2: int, K_2: int)-> float:
    def psf(p, n_1, n_2, W_1, K_1, W_2, K_2):
        p_ans = [0, 0]
        # alpha = calc_alpha_asym(tt, tf, n_1, p[0], n_2, p[1])
        p_ans[0] = p[0] - (1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1))) ) ** (n_1 - 1) *\
                        (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1))) ) ** n_2
        p_ans[1] = p[1] - (1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1))) ) ** n_1 *\
                        (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1))) ) ** (n_2 - 1)
        return p_ans
    p_as = fsolve(psf, [0.4, 0.4], args = (n1, n2, W_1, K_1, W_2, K_2))
    return p_as, np.sqrt(np.sum(np.array(psf(p_as, n1, n2, W_1, K_1, W_2, K_2))**2))

# def calc_as_p_fsolve_iter(n1: int, n2: int, W_1: int, K_1: int, W_2: int, K_2: int)-> float:
#     def psf(p, n_1, n_2, W_1, K_1, W_2, K_2):
#         p_ans = [0, 0]
#         # alpha = calc_alpha_asym(tt, tf, n_1, p[0], n_2, p[1])
#         p_ans[0] = (1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1))) ) ** (n_1 - 1) *\
#                         (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1))) ) ** n_2
#         p_ans[1] =  (1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1))) ) ** n_1 *\
#                         (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1))) ) ** (n_2 - 1)
#         err = (p_ans[0] - p[0]) ** 2 + (p_ans[1] - p[1]) ** 2
#         return p_ans, np.sqrt(err)
#     # us
#     p = [0.4,0.4]
#     for i in range(10000):
#         p, err = psf(p, n1, n2, W_1, K_1, W_2, K_2)
#     p_as = p
#     return p_as, err

if __name__ == "__main__":
    n_range = np.arange(4, 20, 4)
    p1_fsolve1 = []
    p2_fsolve1 = []
    p1_fsolve2 = []
    p2_fsolve2 = []
    for n in n_range:
        lam1 = 0.786 / n
        p_as1, err1 = calc_as_p_fsolve1(n/4, 3*n/4, 8, 6, 16, 6)
        print(f"#1 {n}", p_as1, err1)
        # p_as2, err2 = calc_as_p_fsolve_iter(n/2, n/2, 16, 6, 128, 6)
        # print(f"#2 {n}", p_as2, err2)
        p_as, ss = calc_ss_p_fsolve_pa(n/4, 3*n/4, 8, 6, 16, 6)
        print(f"#3 {n}", p_as[0],p_as[1], ss)
        # p_as1, p_as2, ss = calc_ss_p_formula(n/2, n/2, 16, 6, 128, 6)
        # print(f"#4 {n}", p_as1,p_as2, ss)
        p1_fsolve1.append(p_as1[0])
        p2_fsolve1.append(p_as1[1])
        p1_fsolve2.append(p_as[0])
        p2_fsolve2.append(p_as[1])
    plt.figure(1)
    plt.plot(n_range, p1_fsolve1, label="p1 fsolve1")
    plt.plot(n_range, p2_fsolve1, label="p2 fsolve1")
    plt.plot(n_range, p1_fsolve2, label="p1 formula1")
    plt.plot(n_range, p2_fsolve2, label="p2 formula2")
    plt.legend()
    plt.grid()
    plt.ylabel("p (AS) asym W1 = 16, W2 = 128")
    plt.xlabel("n")
    plt.savefig("p_as_asym.png", dpi=400)
        