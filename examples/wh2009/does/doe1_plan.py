import pandas as pd
import numpy as np
from doepy import build

if __name__ == "__main__":

    np.random.seed(0)

    df_exp = build.full_fact(
        {'lr': [1e-3, 1e-4],
         'epochs': [100, 200],
         'batch_size': [32, 128, 1024],
         'seq_len': [64, 256, 512],
         'est_frac': [0.1, 0.5, 1.0],
         'est_direction': ["forward", "backward"],
         'est_type': ["LSTM", "FF"],
         'est_hidden_size': [8, 16, 32],
         }
    )
    df_exp.index.name = "experiment_id"
    df_exp["seed"] = np.random.permutation(df_exp.index)

    df_exp.to_csv("experiment_plan.csv")

    n_exp = df_exp.shape[0]
    print(f"Planned DOE with {n_exp} runs.")
