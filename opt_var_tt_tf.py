from opt_model import opt_Model
import pandas as pd
data = pd.read_csv("tau_values_new.csv")

if __name__ == "__main__":
    tt1 = data.iloc[3]["tau_t_slots"]
    tf1 = data.iloc[3]['tau_f_slots']
    print(tt1, tf1)
    
    W = 128
    # print(data[["tau_t_slots", "tau_f_slots"]])
    cnt = 0
    df = {"best_beta": [],
          "best_delay(slots)": [],
          "0.5_delay(slots)": [],
          "0.3_delay(slots)": [],
          "0.1_delay(slots)": []}
    for tt2, tf2 in data[["tau_t_slots", "tau_f_slots"]].values:
        cnt += 1
        # print(tt2, tf2)
        model = opt_Model(
        nmld = 20,
        lambda_mld = 0.0005,
        nsld = [20, 20],
        lambda_sld = [0.0001, 0.0001],
        tt = [tt1, tt2],
        tf = [tf1, tf2],
        Wmld= [W, W],
        Wsld= [W, W]
        )
        print(cnt, tt2, tf2, model.opt_delay())
        df["best_beta"].append(model.opt_delay()[0][0])
        df['best_delay(slots)'].append(model.opt_delay()[1])
        df["0.5_delay(slots)"].append(model.delay_of_beta(0.5))
        df["0.3_delay(slots)"].append(model.delay_of_beta(0.3))
        df["0.1_delay(slots)"].append(model.delay_of_beta(0.1))
    dt = pd.DataFrame.from_dict(df)
    data_new = pd.concat([data, dt], axis=1)
    print(data_new)
    data_new.to_csv("tau_values_new_resW128-20.csv", index=False)