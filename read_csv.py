import pandas as pd

df = pd.read_csv("./results/unsym/log-unsym-w16-0.0002-0.0010-0.002000-10-10-10-0.500-32-27.csv")
print(df["p of SLD on Link 2"])