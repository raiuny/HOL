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
    path = "results/log/"
    tt = 32
    tf = 27
    nmld = 10
    nsld1 = nsld2 = 10
    beta = 0.5
    df = None
    for lam1 in np.arange(0.0002, 0.0032, 0.0002):
        lam2 = lam1 / 2
        file = f"log-{lam1:.4f}-{lam2:.4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
        data = pd.read_csv(path+file).drop(columns=["Unnamed: 0"])
        tmp = data.mean().T
        df = pd.concat([df, tmp], axis=1)
    df = df.T
    return df


if __name__ == "__main__":
    n1 = 10
    n2 = 10
    tau_T = 32
    tau_F = 27
    lam_range = np.arange(0.0002, 0.0032, 0.0002)
    thpt_res = []
    ac_delay_res = []
    q_delay_res = []
    e2e_delay_res = []
    p_res = []
    alpha_res1 = []
    alpha_sim_calc = []
    for lam1 in lam_range:
        model = HOL_Model(
            n1 = 10,
            n2 = 10,
            lambda1 = tau_T * lam1/2,
            lambda2 = tau_T * lam1/2,
            W_1 = 16,
            W_2 = 16,
            K_1 = 6,
            K_2 = 6,
            tt = tau_T,
            tf = tau_F
            )
        p_res.append(np.mean(model.p))
        thpt_res.append(np.sum(model.throughput_Mbps) * 2)
        ac_delay_res.append(np.mean(model.access_delay_ms))
        q_delay_res.append(np.mean(model.queuing_delay_ms))
        e2e_delay_res.append(np.mean(model.e2e_delay_ms))
        alpha_res1.append(model.alpha)
    sim = get_result_sim()
    print(sim)
    

    
    
    
    
    # ns-3
    
    # Coex data
    coex_tp = [5.3492, 10.5516, 15.8524, 21.144, 26.478, 31.7816, 37.066, 42.352, 47.7192, 53.1012, 58.3768, 58.1176, 58.0488, 58.0616, 58.1292]
    coex_qd = [0.000406154, 0.00105731, 0.00186477, 0.00338285, 0.00612223, 0.00983223, 0.0174359, 0.0345943, 0.0799976, 0.192421, 1.39091, 1491.22, 2930.55, 4106.83, 5100.39]
    coex_ad = [0.274535, 0.292121, 0.313937, 0.341613, 0.375804, 0.418707, 0.477762, 0.561661, 0.700434, 0.926548, 1.6046, 8.13827, 8.41872, 8.48173, 8.54391]
    coex_ed = [0.274941, 0.293178, 0.315801, 0.344996, 0.381926, 0.42854, 0.495198, 0.596255, 0.780432, 1.11897, 2.9955, 1499.36, 2938.96, 4115.31, 5108.93]
    
    plt.figure(1)
    plt.plot(lam_range, (sim["Throughput of MLD on Link 1"] + sim["Throughput of MLD on Link 2"] + sim["Throughput of SLD on Link 1"] + sim["Throughput of SLD on Link 2"] )  * 1500 * 8 / 9, label="sim-py")
    plt.plot(lam_range, thpt_res, label="model", linestyle="--")
    plt.plot(lam_range, coex_tp, label="sim-ns3", linestyle = "--")
    plt.ylabel("Throughput(packet per slot) (Mbps)")
    plt.xlabel("arrival rate of mld")
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.plot(lam_range, (sim['Access Delay of MLD on Link 1'] + sim['Access Delay of MLD on Link 2'] + sim['Access Delay of SLD on Link 1'] + sim['Access Delay of SLD on Link 1']) / 4  * 9 * 1e-3, label="sim")
    plt.plot(lam_range, ac_delay_res, label = "model", linestyle="--")
    plt.plot(lam_range, coex_ad, label = "sim-ns3", linestyle="--")
    print("ac delay", ac_delay_res)
    plt.ylabel("Access Delay (ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(3)
    plt.plot(lam_range[:-5], sim['Queueing Delay of MLD on Link 1'][:-5] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:-5], q_delay_res[:-5], label = "model", linestyle="--")
    plt.plot(lam_range[:-5], coex_qd[:-5], label = "sim-ns3", linestyle="--")
    
    print("q delay", q_delay_res)
    plt.ylabel("Queueing Delay (ms)")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(4)
    plt.plot(lam_range[:-5], sim['weighted e2e delay of mld'][:-5] * 9 * 1e-3, label="sim")
    plt.plot(lam_range[:-5], e2e_delay_res[:-5], label = "model", linestyle="--")
    plt.plot(lam_range[:-5], coex_ed[:-5], label = "sim-ns3", linestyle="--")
    print("e2e delay", q_delay_res)
    plt.ylabel("E2E Delay (ms)")
    plt.xlabel("arrival rate of mld")
    plt.grid()
    plt.legend()
    plt.figure(5)
    plt.scatter(lam_range, sim['p of SLD on Link 1'], label="sim")
    plt.plot(lam_range, p_res, label = "model", linestyle="--")
    # plt.scatter(lam_range, p_res, label = "model")
    print("p", p_res)
    plt.ylabel("p")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.figure(6)
    plt.scatter(lam_range, sim['alpha of link 1'], label="sim")
    plt.plot(lam_range, alpha_res1, label = "model", linestyle="--")
    # plt.scatter(lam_range, p_res, label = "model")
    print("alpha1", alpha_res1)
    plt.ylabel("alpha")
    plt.xlabel("arrival rate of mld")
    plt.legend()
    plt.grid()
    plt.show()
    