import os
import numpy as np
import torch
import pandas as pd
from torchid.ss.dt.simulator import StateSpaceSimulator
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from examples.pick_place.loader import pick_place_loader, pick_place_scaling
from torchid import metrics


if __name__ == '__main__':

    DOE_NAME = "doe1"
    df_plan = pd.read_csv(os.path.join("does", f"{DOE_NAME}_plan.csv"))  # doe1_plan.csv
    df_plan.set_index("experiment_id", inplace=True)

    decimate = 10

    df_res = df_plan.copy()
    df_res["FIT"] = np.nan
    df_res["RMSE"] = np.nan

    # Load dataset
    t, u, y = pick_place_loader("test", scale=True)
    u_mean, u_std, y_mean, y_std = pick_place_scaling()

    torch.set_num_threads(4)

    u_v = torch.from_numpy(u[:, None, :])
    y_v = torch.from_numpy(y[:, None, :])

    y = y * y_std + y_mean  # scaled, for statistics

    cnt = 0
    for exp_id, row in df_plan.iterrows():
        print(f"Evaluating experiment {exp_id}...")
        filename = os.path.join("models", DOE_NAME, f"model_{exp_id:.0f}.pt")
        if not os.path.exists(filename):
            continue
        model_data = torch.load(filename)

        n_x = model_data["n_x"]
        n_y = model_data["n_y"]
        n_u = model_data["n_u"]
        args = model_data["args"]

        f_xu = models.NeuralLinStateUpdateV2(n_x, n_u, hidden_size=args.hidden_size)
        g_x = models.LinearOutput(n_x, n_u)
        model = StateSpaceSimulator(f_xu, g_x)

        if args.est_type == "LSTM":
            estimator = estimators.LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                      hidden_size=args.est_hidden_size,
                                                      flipped=False)
        elif args.est_type == "FF":
            estimator = estimators.FeedForwardStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                             hidden_size=args.est_hidden_size,
                                                             seq_len=args.seq_est_len)
        elif args.est_type == "ZERO":
            estimator = estimators.ZeroStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x)
        elif args.est_type == "RAND":
            estimator = estimators.RandomStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x)
        else:
            raise ValueError("Wrong estimator type. Possible values: LSTM|FF|ZERO|RAND")

        model.load_state_dict(model_data["model"])
        estimator.load_state_dict(model_data["estimator"])

        with torch.no_grad():
            model.eval()
            estimator.eval()

            u_est = u_v[:args.seq_est_len]
            y_est = y_v[:args.seq_est_len]
            x0 = estimator(u_est, y_est)

            if args.est_type not in ["ZERO", "RAND"]:  # for not-dummy estimators
                u_fit = u_v[args.seq_est_len:]
            else:
                u_fit = u_v

            y_sim = model(x0, u_fit).squeeze(1).detach().numpy()
            y_sim = np.r_[np.zeros((args.seq_est_len, 1)), y_sim]

    df_res.to_csv(os.path.join("does", DOE_NAME + "_res.csv"))
