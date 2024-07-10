from hol_model import HOL_Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from utils import calc_alpha_sym
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# cmap = mpl.cm.get_cmap('viridis')


def get_result_sim():
    path = "../simulation_python/log/"
    tt = 32
    tf = 27
    nmld = 10
    beta = 0.5
    # df = None
    lam1 = 0.002
    lam21 = 0.001
    lam22 = 0.001
    best_beta = []
    best_delay = []
    for nsld1 in np.arange(0, 21, 1):
        nsld2 = 20 - nsld1
        best_beta_tmp = 0
        min_delay = 1e9
        for beta in np.arange(0.40, 0.61, 0.01):
            file = f"log-var-beta-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
            data = pd.read_csv(path+file).drop(columns=["Unnamed: 0"])
            tmp = data.mean().T
            if tmp["weighted e2e delay of mld"] < min_delay:
                min_delay = tmp["weighted e2e delay of mld"]
                best_beta_tmp = beta
            # df = pd.concat([df, tmp], axis=1)
        best_beta.append(best_beta_tmp)
        best_delay.append(min_delay)
    print(best_beta, best_delay)
    return best_beta


if __name__ == "__main__":
    n1 = 10
    n2 = 10
    tau_T = 32
    tau_F = 27
    lam_range = np.arange(0.0002, 0.0034, 0.0002)
    thpt1_res = []
    thpt2_res = []
    ac_delay_res1 = []
    ac_delay_res2 = []
    q_delay_res1 = []
    q_delay_res2 = []
    e2e_delay_res = []
    # e2e_delay_res2 = []
    p1_res = []
    p2_res = []
    alpha_res1 = []
    alpha_res2 = []
    alpha_sim_calc = []
    states = []
    lam1 = 0.002
    lam21 = lam22 = 0.001
    best_beta = []
    best_delay = []
    for nsld1 in np.arange(0, 21, 1):
        best_beta_tmp = -1
        best_delay_tmp = 1e20
        for beta in np.arange(0.4, 1.00, 0.01):
            model1 = HOL_Model(
                n1 = 10,
                n2 = nsld1,
                lambda1 = tau_T * lam1, 
                lambda2 = tau_T * lam21,
                W_1 = 16,
                W_2 = 16,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            model2 = HOL_Model(
                n1 = 10,
                n2 = 20 - nsld1,
                lambda1 = tau_T * lam1,
                lambda2 = tau_T * lam22,
                W_1 = 16,
                W_2 = 16,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            if model1.e2e_delay[0] * beta + (1 - beta) * model2.e2e_delay[0] < best_delay_tmp:
                best_delay_tmp = model1.e2e_delay[0] * beta + (1 - beta) * model2.e2e_delay[0]
                best_beta_tmp = beta
        best_beta.append(best_beta_tmp)
        best_delay.append(best_delay_tmp)
    print("model", best_beta)
    sim = get_result_sim()
    print("sim", sim)
    exit()
    print(states)
    
    
    
    
    # ns-3
    
    
    plt.figure(1)
    plt.plot(lam_range, sim["Throughput of MLD on Link 1"] * 1500 * 8 / 9, label="sim-py")
    plt.plot(lam_range, thpt1_res, label="model", linestyle="--")
    plt.ylabel("Throughput of MLD on Link 1(Mbps)")
    plt.xlabel("arrival rate of mld")
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.plot(lam_range, sim["Throughput of MLD on Link 2"]  * 1500 * 8 / 9, label="sim-py")
    plt.plot(lam_range, thpt2_res, label="model", linestyle="--")
    plt.ylabel("Throughput of MLD on Link 2(Mbps)")
    plt.xlabel("arrival rate of mld")
    plt.grid()
    plt.legend()
    plt.figure(3)
    plt.plot(lam_range, sim['Access Delay of MLD on Link 1']  * 9 * 1e-3, label="sim")
    plt.plot(lam_range, ac_delay_res1, label = "model", linestyle="--")
    print("ac delay1", ac_delay_res1)
    plt.ylabel("Access Delay of MLD on Link 1(ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(4)
    plt.plot(lam_range, sim['Access Delay of MLD on Link 2']  * 9 * 1e-3, label="sim")
    plt.plot(lam_range, ac_delay_res2, label = "model", linestyle="--")
    print("ac delay2", ac_delay_res2)
    plt.ylabel("Access Delay of MLD on Link 2(ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(5)
    plt.plot(lam_range[:-3], sim['Queueing Delay of MLD on Link 1'][:-3] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:-3], q_delay_res1[:-3], label = "model", linestyle="--")
    print("q delay1", q_delay_res1)
    plt.ylabel("Queueing Delay of MLD on Link 1(ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(6)
    plt.plot(lam_range[:-3], sim['Queueing Delay of MLD on Link 2'][:-3] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:-3], q_delay_res2[:-3], label = "model", linestyle="--")
    print("q delay2", q_delay_res2)
    plt.ylabel("Queueing Delay of MLD on Link 2(ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(7)
    plt.plot(lam_range[:-3], sim['weighted e2e delay of mld'][:-3] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:-3], e2e_delay_res[:-3], label = "model", linestyle="--")
    plt.ylabel("E2E Delay of MLD(ms)")
    plt.xlabel("arrival rate of mld")
    plt.grid()
    plt.legend()
    plt.figure(8)
    plt.scatter(lam_range, sim['p of MLD on Link 1'], label="sim")
    plt.plot(lam_range, p1_res, label = "model", linestyle="--")
    # plt.scatter(lam_range, p_res, label = "model")
    plt.ylabel("p of MLD on Link 1")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(9)
    plt.scatter(lam_range, sim['p of MLD on Link 2'], label="sim")
    plt.plot(lam_range, p2_res, label = "model", linestyle="--")
    # plt.scatter(lam_range, p_res, label = "model")
    plt.ylabel("p of MLD on Link 2")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(10)
    plt.scatter(lam_range, sim['alpha of link 1'], label="sim")
    plt.plot(lam_range, alpha_res1, label = "model", linestyle="--")
    plt.ylabel("alpha of Link 1")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(11)
    plt.scatter(lam_range, sim['alpha of link 2'], label="sim")
    plt.plot(lam_range, alpha_res2, label = "model", linestyle="--")
    plt.ylabel("alpha of Link 2")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.show()
    