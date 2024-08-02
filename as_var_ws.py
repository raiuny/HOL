from hol_model import HOL_Model
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'figure.dpi': 400,
    'savefig.dpi': 400
})
tau_T = 32
tau_F = 27
n1 = 10
n2 = 10
lam1 = 0.005
lam2 = 0.005
mld_delay = {}
sld_delay = {}
for wm in [16, 32, 64]:
    mld_delay[wm] = []
    sld_delay[wm] = []
    for ws in [16, 32, 64, 128,256, 512, 1024]:
        model = HOL_Model(
                n1 = n1,
                n2 = n2,
                lambda1 = tau_T * lam1,
                lambda2 = tau_T * lam2,
                W_1 = wm,
                W_2 = ws,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
            )
        # assert model.state == "SS", f"{wm}, {ws}"
        mld_delay[wm].append(model.access_delay_ms[0])
        sld_delay[wm].append(model.access_delay_ms[1])

plt.figure(1)
for wm in [16, 32, 64]:
    plt.plot([16, 32, 64, 128,256, 512,1024], mld_delay[wm], label = f"AD1 Wm = {wm}")
    plt.plot([16, 32, 64, 128, 256, 512, 1024], sld_delay[wm], label = f"AD2 Wm = {wm}")
plt.xlabel("Ws")
plt.ylabel("access delay AS")
plt.xscale("log")
plt.xticks([16, 32, 64, 128, 256, 512, 1024], labels=[16, 32, 64, 128, 256, 512, 1024])
plt.yscale("log")
plt.legend()
plt.savefig("ad_vs_ws.png")
mld_delay = []
sld_delay = []
for wm in np.arange(16, 1024, 8):
    ws = wm
    model = HOL_Model(
            n1 = n1,
            n2 = n2,
            lambda1 = tau_T * lam1,
            lambda2 = tau_T * lam2,
            W_1 = wm,
            W_2 = ws,
            K_1 = 6,
            K_2 = 6,
            tt = tau_T,
            tf = tau_F
        )
    # assert model.state == "SS", f"{wm}, {ws}"
    mld_delay.append(model.access_delay_ms[0])
    sld_delay.append(model.access_delay_ms[1])

plt.figure(2)
plt.plot( np.arange(16, 1024, 8), mld_delay, label = f"AD1")
plt.plot( np.arange(16, 1024, 8), sld_delay, label = f"AD2")
plt.xscale("log")
plt.xticks([16, 32, 64, 128, 256, 512, 1024], labels=[16, 32, 64, 128, 256, 512, 1024])
plt.ylabel("access delay AS")
plt.xlabel("Ws")
plt.legend()
plt.savefig("ad_vs_ws2.png")