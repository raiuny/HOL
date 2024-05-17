from hol_model import HOL_Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
cmap = mpl.cm.get_cmap('viridis')

def plot_model_fig(x, y, label, opt_mark = "x" , color = "green", num = 0):
    idx = np.argmin(y)
    pltx, plty = [], []
    for xi, yi in zip(x, y):
        if yi < 2000:
            pltx.append(xi)
            plty.append(yi)
    plt.plot(x[idx], y[idx], marker = opt_mark, color = color)
    plt.scatter(pltx, plty, label=label, marker=".", color = color)
    plt.annotate(f'({x[idx]:.3f}, {y[idx]:.1f})',  # 注释内容为坐标
             xy=(x[idx], y[idx]),  # 指定注释的位置（即点的坐标）
             textcoords="offset points",  # 设置文本相对数据点的偏移方式
             xytext=(0,-8),  # 设置文本相对点的偏移量，这里表示在点右上方10个单位
             ha='center', color=color, size=8)  # 水平居中对齐
    plt.plot(pltx, plty, label=label, linestyle="--", color = color)
    # plt.scatter(x, y)
    
def plot_sim_fig(x, y, label, opt_mark = "^", color = "green"):
    idx = np.argmin(y)
    pltx, plty = [], []
    for xi, yi in zip(x, y):
        if yi < 600:
            pltx.append(xi)
            plty.append(yi)
    plt.plot(x[idx], y[idx], marker = opt_mark, color = color)
    plt.scatter(pltx, plty, label=label, marker=".", color = color)
    plt.annotate(f'({x[idx]:.2f}, {y[idx]:.1f})',  # 注释内容为坐标
             xy=(x[idx], y[idx]),  # 指定注释的位置（即点的坐标）
             textcoords="offset points",  # 设置文本相对数据点的偏移方式
             xytext=(0,+8),  # 设置文本相对点的偏移量，这里表示在点右上方10个单位
             ha='center', color=color, size=8)  # 水平居中对齐
    plt.plot(pltx, plty, label=label, linestyle="-", color = color)
    
if __name__ == "__main__":
    tau_T = 36
    tau_F = 28
    nmld = 20
    result = {}
    states = {}
    arrival_rate_mld = 0.001
    arrival_rate_sld = 0.0002
    beta_range = np.arange(0.1, 1.0, 0.01)
    nsld1_range = [0, 4, 10]
    for nsld1 in nsld1_range:
        nsld2 = 20 - nsld1
        result[nsld1]=[]
        states[nsld1]=[]
        for beta in beta_range:
            # link1
            model1 = HOL_Model(
                n1 = nmld,
                n2 = nsld1,
                lambda1 = tau_T * arrival_rate_mld * beta,
                lambda2 = tau_T * arrival_rate_sld,
                W_1 = 128,
                W_2 = 128,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            model2 = HOL_Model(
                n1 = nmld,
                n2 = nsld2,
                lambda1 = tau_T * arrival_rate_mld * (1-beta),
                lambda2 = tau_T * arrival_rate_sld,
                W_1 = 128,
                W_2 = 128,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            result[nsld1].append(model1.e2e_delay[0] * beta + model2.e2e_delay[0] * (1-beta))
            states[nsld1].append(model1.state + '|' + model2.state)
    for k, v in result.items():
        print(k, v)
    for k, v in states.items():
        print(k, v)
    
    for i, c in zip(nsld1_range, ["g", "r", "b"]):
        if i== 4:
            plot_model_fig(beta_range, result[i], label=f"model {i}:{20-i}", color=c)
    import pandas as pd
    sim_res = pd.read_csv("result/var_beta_sld_4_16_W128.csv")
    plot_sim_fig(beta_range, sim_res["E2E Delay(python sim)"], label=f"sim {4}:{16}")
    plt.ylabel("E2E delay")
    plt.xlabel(r"$\beta$")
    plt.legend()
    plt.savefig("var_beta_lam1_0.001_lam2_0.0002_nmld_20_sld_4_16_W128.png")
    plt.show()