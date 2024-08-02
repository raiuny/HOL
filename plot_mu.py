from utils import calc_mu_S, calc_mu_U, calc_au_p_fsolve, calc_as_p_fsolve
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'figure.dpi': 400,
    'savefig.dpi': 400
})
tt = 32
tf = 27
W = 16
K = 6
mu_s_res = []
mu_u_res = []
rho = []
lam_range = np.arange(0.01, 1.01, 0.01)
for lam1 in lam_range:
    p, flag = calc_au_p_fsolve(10, 10, lam1/20, lam1/20, tt, tf)
    p_uu = p[0]
    if flag:
        mu_S = min(calc_mu_S(p_uu, tt, tf, W, K), 1)
        mu_U = calc_mu_U(p_uu, tt, tf, W, K, lam1) 
        rho.append(lam1 / mu_U / 20)
    else:
        p_ss, flag = calc_as_p_fsolve(10, 10, W, K, W, K)
        mu_S = calc_mu_S(p_ss[0], tt, tf, W, K)
        mu_U = calc_mu_U(p_ss[0], tt, tf, W, K, lam1) 
        rho.append(1)
    mu_s_res.append(mu_S)
    mu_u_res.append(mu_U)

plt.figure(1)
plt.plot(lam_range, mu_s_res, label=r"$\mu_S$")
plt.plot(lam_range, mu_u_res, label=r"$\mu_U$")
plt.plot(lam_range,lam_range/20, label=r"$\lambda$")
plt.grid()
plt.legend()
plt.xlabel(r"$\hat{\lambda}$")
plt.ylabel(r"$\mu$")
plt.savefig("mu.png")
plt.figure(2)
plt.plot(lam_range, rho, label=r"$\rho$")
plt.ylabel(r"$\rho$")
plt.xlabel(r"$\hat{\lambda}$")
plt.legend()
plt.savefig("rho.png")
