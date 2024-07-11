from utils import calc_uu_p_formula, calc_uu_p_fsolve, calc_ps_p_formula, calc_ps_p_fsolve, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_fsolve_iter
import matplotlib.pyplot as plt
import numpy as np

tt = 32
tf = 27

# au
lam_u = 0.1
p_formula = []
p_fsolve = []
n_range = range(10, 110, 10)
for n in n_range:
    lam1 = 0.1 / n 
    p_uu1, p_uu2, flag1 = calc_uu_p_formula(n/2, lam1, n/2, lam1, tt, tf)
    print(f"#1 {n}", p_uu1, p_uu2, flag1)
    p_uu3, p_uu4, flag2 = calc_uu_p_fsolve(n/2, lam1, n/2, lam1, tt, tf)
    print(f"#2 {n}", p_uu3, p_uu4, flag2)
    p_formula.append(p_uu1)
    p_fsolve.append(p_uu3)
plt.figure(1)
plt.plot(n_range, p_formula, label = "formula")
plt.plot(n_range, p_fsolve, label = "fsolve")
plt.legend()
plt.grid()
plt.xlabel("n")
plt.ylabel("p (AU) SYM")
plt.yticks(np.arange(0.99, 1.005, 0.005))
plt.savefig("p_uu.png", dpi=400)

# as
lam_as = 1
p_ss_formula = []
p_ss_fsolve = []
for n in n_range:
    lam1 = lam_as / n 
    p_ss1, p_ss2, flag1 = calc_ss_p_formula(n/2, n/2, 16, 6, 16, 6)
    print(f"#1 {n}", p_ss1, p_ss2, flag1)
    p_ss3, p_ss4, flag2 = calc_ss_p_fsolve(n/2, n/2, 16, 6, 16, 6)
    print(f"#2 {n}", p_ss3, p_ss4, flag2)
    p_ss_formula.append(p_ss1)
    p_ss_fsolve.append(p_ss3)

plt.figure(2)
plt.plot(n_range, p_ss_formula, label = "formula")
plt.plot(n_range, p_ss_fsolve, label = "fsolve")
plt.legend()
plt.grid()
plt.xlabel("n")
plt.ylabel("p (AS) K = 6 SYM")
plt.savefig("p_as.png", dpi=400)

# ps 
p1_us_formula = []
p2_us_formula = []
p1_us_fsolve = []
p2_us_fsolve = []
for n in n_range:
    lam1 = 0.1 / n * 2
    lam2 = 0.8 / n * 2
    p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n/2, lam1, n/2, lam2, 16, 6, 16, 6, tt, tf)
    print(f"#1 {n}", p_us1, p_us2, p_su1, p_su2)
    p1_us_formula.append(p_us1)
    p2_us_formula.append(p_us2)
    p_us, p_su, err1, err2= calc_ps_p_fsolve(n/2, lam1, n/2, lam2, 16, 6, 16, 6, tt, tf)
    print(f"#2 {n}", p_us, p_su, err1, err2)
    p1_us_fsolve.append(p_us[0])
    p2_us_fsolve.append(p_us[1])
plt.figure(3)
plt.plot(n_range, p1_us_formula, label = "p1 formula")
plt.plot(n_range, p2_us_formula, label = "p2 formula")
plt.plot(n_range, p1_us_fsolve, label = "p1 fsolve")
plt.plot(n_range, p2_us_fsolve, label = "p2 fsolve")
plt.legend()
plt.grid()
plt.ylabel("p (US) SYM")
plt.xlabel("n")
plt.savefig("p_us.png", dpi=400)

p1_us_formula = []
p2_us_formula = []
p1_us_fsolve = []
p2_us_fsolve = []
for n in n_range:
    lam1 = 0.1 / n * 2
    lam2 = 0.8 / n * 2
    p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n/2, lam1, n/2, lam2, 16, 6, 128, 6, tt, tf)
    print(f"#1 {n}", p_us1, p_us2, p_su1, p_su2)
    p1_us_formula.append(p_us1)
    p2_us_formula.append(p_us2)
    p_us, p_su, err1, err2= calc_ps_p_fsolve(n/2, lam1, n/2, lam2, 16, 6, 128, 6, tt, tf)
    print(f"#2 {n}", p_us, p_su, err1, err2)
    p1_us_fsolve.append(p_us[0])
    p2_us_fsolve.append(p_us[1])
plt.figure(4)
plt.plot(n_range, p1_us_formula, label = "p1 formula")
plt.plot(n_range, p2_us_formula, label = "p2 formula")
plt.plot(n_range, p1_us_fsolve, label = "p1 fsolve")
plt.plot(n_range, p2_us_fsolve, label = "p2 fsolve")
plt.legend()
plt.grid()
plt.ylabel("p (US) ASYM W1,W2=16,128")
plt.xlabel("n")
plt.savefig("p_us_asym.png", dpi=400)
    
    