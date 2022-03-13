import pandas as pd
import numpy as np
from doepy import build

if __name__ == "__main__":

    np.random.seed(0)

    df_exp = build.full_fact(
        {'lr': [1e-4, 1e-3],
         'epochs': [10000],
         'max_time': [300, 1800],
         'batch_size': [128, 1024],
         'seq_len': [40, 80, 256],
         'seq_est_len': [6, 50, 100],
         'est_direction': ["forward"],
         'est_type': ["LSTM", "FF", "ZERO"],
         'est_hidden_size': [15],
         }
    )
    df_exp.index.name = "experiment_id"
    df_exp["seed"] = np.random.permutation(df_exp.index)

    df_exp = df_exp.sample(frac=1.0)
    df_exp.to_csv("doe3_plan.csv")

    n_exp = df_exp.shape[0]
    print(f"Planned DOE with {n_exp} runs.")
    est_time = df_exp["max_time"].sum() / 3600
    print(f"Required time: {est_time} hours")
