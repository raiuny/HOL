import pandas as pd
import numpy as np

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

print(df)
df = df.T
print(list((df["Throughput of MLD on Link 2"]+df["Throughput of MLD on Link 1"])*2))