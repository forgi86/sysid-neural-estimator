import os
import numpy as np
import torch
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import pick_place_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #model_data = torch.load(os.path.join("models", "model.pt"), map_location=torch.device('cpu'))
    model_data = torch.load(os.path.join("models", "doe1", "model_129.pt"), map_location=torch.device('cpu'))

    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    args = model_data["args"]
    decimate = 10

    t_full, u_full, y_full = pick_place_loader(dataset="full", decimate=decimate, scale=True)

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
        u_v = torch.tensor(u_full[:, None, :])
        y_v = torch.tensor(y_full[:, None, :])

        u_est = u_v[:args.seq_est_len]
        y_est = y_v[:args.seq_est_len]
        x0 = estimator(u_est, y_est)

        if args.est_type not in ["ZERO", "RAND"]:  # for not-dummy estimators
            u_fit = u_v[args.seq_est_len:]
        else:
            u_fit = u_v

        y_sim = model(x0, u_fit).squeeze(1).detach().numpy()

        if args.est_type not in ["ZERO", "RAND"]:  # for non-dummy estimators
            y_sim = np.r_[np.zeros((args.seq_est_len, 1)), y_sim]

    # %% Metrics

    from torchid import metrics
    e_rms = metrics.rmse(y_full, y_sim)[0]
    fit_idx = metrics.fit_index(y_full, y_sim)[0]
    r_sq = metrics.r_squared(y_full, y_sim)[0]

    print(f"RMSE: {e_rms:.1f} \nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

    # %% Plots
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_full[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label='sim')
