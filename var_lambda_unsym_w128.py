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
cmap = mpl.cm.get_cmap('viridis')


def get_result_sim():
    path = "results/unsym/"
    tt = 32
    tf = 27
    nmld = 10
    nsld1 = nsld2 = 10
    beta = 0.5
    df = None
    for lam1 in np.arange(0.0002, 0.0034, 0.0002):
        lam21 = 0.0010
        lam22 = 0.0020
        file = f"log-unsym-w128-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
        data = pd.read_csv(path+file).drop(columns=["Unnamed: 0"])
        tmp = data.mean().T
        df = pd.concat([df, tmp], axis=1)
    df = df.T
    print(df["p of MLD on Link 2"])
    return df


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
    for lam1 in lam_range:
        model1 = HOL_Model(
            n1 = 10,
            n2 = 10,
            lambda1 = tau_T * lam1/2,
            lambda2 = tau_T * 0.0010,
            W_1 = 128,
            W_2 = 16,
            K_1 = 6,
            K_2 = 6,
            tt = tau_T,
            tf = tau_F
            )
        model2 = HOL_Model(
            n1 = 10,
            n2 = 10,
            lambda1 = tau_T * lam1/2,
            lambda2 = tau_T * 0.0020,
            W_1 = 128,
            W_2 = 16,
            K_1 = 6,
            K_2 = 6,
            tt = tau_T,
            tf = tau_F
            )
        states.append(model1.state+model2.state)
        p1_res.append(np.mean(model1.p1))
        p2_res.append(np.mean(model2.p1))
        thpt1_res.append(model1.throughput_Mbps[0])
        thpt2_res.append(model2.throughput_Mbps[0])
        ac_delay_res1.append(model1.access_delay_ms[0])
        ac_delay_res2.append(model2.access_delay_ms[0])
        q_delay_res1.append(model1.queuing_delay_ms[0])
        q_delay_res2.append(model2.queuing_delay_ms[0])
        e2e_delay_res.append(0.5 * model1.e2e_delay_ms[0] + 0.5 * model2.e2e_delay_ms[0])
        # e2e_delay_res2.append(model2.e2e_delay_ms[0])
        alpha_res1.append(model1.alpha)
        alpha_res2.append(model2.alpha)
    sim = get_result_sim()
    print(sim)
    
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
    plt.plot(lam_range[:4], sim['Queueing Delay of MLD on Link 2'][:4] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:4], q_delay_res2[:4], label = "model", linestyle="--")
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
    