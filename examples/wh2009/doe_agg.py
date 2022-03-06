import os
import numpy as np
import torch
import pandas as pd
from torchid.ss.dt.simulator import StateSpaceSimulator
import torchid.ss.dt.models as models
from examples.wh2009.loader import wh2009_loader, wh2009_scaling
from torchid import metrics


if __name__ == '__main__':

    DOE_NAME = "doe1"
    df_plan = pd.read_csv(os.path.join("does", f"{DOE_NAME}_plan.csv"))  # doe1_plan.csv
    df_plan.set_index("experiment_id", inplace=True)

    df_res = df_plan.copy()
    df_res["FIT"] = np.nan
    df_res["RMSE"] = np.nan

    # Load dataset
    t, u, y = wh2009_loader("test", scale=True)
    y_mean, y_std = wh2009_scaling()
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

        f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=args.hidden_size)
        g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size)  #LinearOutput(n_x, n_y)
        model = StateSpaceSimulator(f_xu, g_x)
        model.load_state_dict(model_data["model"])

        with torch.no_grad():
            # x0 = estimator(u_v, y_v)
            x0 = torch.zeros((1, n_x), dtype=u_v.dtype, device=u_v.device)  # initial state set to 0 for simplicity
            y_sim = model(x0, u_v).squeeze(1)  # remove batch dimension
        y_sim = y_sim.detach().numpy()

        y_sim = y_sim*y_std + y_mean

        e_rms = 1000 * metrics.rmse(y, y_sim)[0]
        fit_idx = metrics.fit_index(y, y_sim)[0]
        r_sq = metrics.r_squared(y, y_sim)[0]

        df_res.loc[exp_id, "FIT"] = fit_idx
        df_res.loc[exp_id, "RMSE"] = e_rms

        #cnt += 1
        #if cnt == 5:
        #    break
    df_res.to_csv("does", DOE_NAME + "_res.csv")
