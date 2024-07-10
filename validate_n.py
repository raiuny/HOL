import pandas as pd
import numpy as np
from hol_model import HOL_Model
import matplotlib.pyplot as plt
tt = 32
tf = 27
nsld2 = 0
lam1 = 0
lam21 = 0.005
lam22 = 0
nmld = 0
beta = 0.5
tpt_sim = []
tpt_model = []
for nsld1 in np.arange(10, 110, 10):
    file  = f"./results/log-n/log-var-nsld1-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
    data = pd.read_csv(file)
    tpt_sim.append(data.mean()["Throughput of SLD on Link 1"])
    model = HOL_Model(
        n1 = nsld1,
        n2 = 0,
        lambda1 = tt * lam21,
        lambda2 = tt * lam21,
        W_1 = 16,
        W_2 = 16,
        K_1 = 6,
        K_2 = 6,
        tt = tt,
        tf = tf
    )
    tpt_model.append(model.throughput[0])
print(tpt_sim, tpt_model)
plt.plot(np.arange(10, 110, 10), tpt_sim, label = "sim")
plt.plot(np.arange(10, 110, 10), tpt_model, label="model")
plt.plot(np.arange(10, 110, 10), [0.023789333333333298, 0.0222741333333333, 0.02133519999999998, 0.020574399999999982, 0.01997946666666664, 0.019555733333333318, 0.01903679999999996, 0.018680533333333298, 0.0182781333333333, 0.017981599999999962] , label="sim-bf")
plt.legend()
plt.grid()
plt.show()