from opt_model import opt_Model
import pandas as pd
data = pd.read_csv("tau_values_new.csv")

if __name__ == "__main__":
    tt1 = data.iloc[12]["tau_t_slots"]
    tf1 = data.iloc[12]['tau_f_slots']
    print(tt1, tf1)
    
    W = 128
    # print(data[["tau_t_slots", "tau_f_slots"]])
    cnt = 0
    df = {
        "lambda_s1/lambda_s2": [],
        "best_beta": [],
        "best_delay(slots)": [],
        "0.5_delay(slots)": [],
        "0.3_delay(slots)": [],
        "0.1_delay(slots)": []}
    n_range = range(5, 105, 5)
    for r in n_range:
        cnt += 1
        # print(tt2, tf2)
        model = opt_Model(
        nmld = 20,
        lambda_mld = 0.0005,
        nsld = [20, 20],
        lambda_sld = [0.00001 *  r, 0.00001],
        tt = [tt1, tt1],
        tf = [tf1, tf1],
        Wmld= [W, W],
        Wsld= [W, W]
        )

        print(cnt, r, model.opt_delay())
        df["lambda_s1/lambda_s2"].append(r)
        df["best_beta"].append(model.opt_delay()[0][0])
        df['best_delay(slots)'].append(model.opt_delay()[1])
        df["0.5_delay(slots)"].append(model.delay_of_beta(0.5))
        df["0.3_delay(slots)"].append(model.delay_of_beta(0.3))
        df["0.1_delay(slots)"].append(model.delay_of_beta(0.1))
    dt = pd.DataFrame.from_dict(df)
    print(dt)
    dt.to_csv("var_nsld1_beta_res.csv")
    # data_new = pd.concat([data, dt], axis=1)
    # print(data_new)
    # data_new.to_csv("tau_values_new_res.csv", index=False)