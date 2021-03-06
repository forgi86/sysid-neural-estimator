import os
import torch
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator
from loader import wh2009_loader, wh2009_scaling
import matplotlib
import matplotlib.pyplot as plt
from torchid import metrics


if __name__ == '__main__':

    matplotlib.rc('font', **{'size': 14, 'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    model_data = torch.load(os.path.join("models", "doe5", "model_113.pt"), map_location=torch.device('cpu'))  # best

    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    args = model_data["args"]

    # Load dataset
    t, u, y = wh2009_loader("test", scale=True)
    y_mean, y_std = wh2009_scaling()

    #%% Load models and parameters
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=args.hidden_size)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size)  #LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    estimator = LSTMStateEstimator(n_u=n_u, n_y=n_y, n_x=n_x, flipped=True)
    model.load_state_dict(model_data["model"])
    #state_estimator.load_state_dict(model_data["estimator"])

    #%% Simulate
    with torch.no_grad():
        u_v = torch.tensor(u[:, None, :])
        y_v = torch.tensor(y[:, None, :])
        # x0 = estimator(u_v, y_v)
        # initial state here set to 0 for simplicity. The effect on the long simulation is negligible
        x0 = torch.zeros((1, n_x), dtype=u_v.dtype, device=u_v.device)
        y_sim = model(x0, u_v).squeeze(1)  # remove batch dimension
    y_sim = y_sim.detach().numpy()

    y = y*y_std + y_mean
    y_sim = y_sim*y_std + y_mean

    #%% Test
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y[:, 0], 'k', label='$y$')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label=r'$y^{\rm sim}$')
    ax.plot(y[:, 0] - y_sim[:, 0], 'r', label=r'$y-y^{\rm sim}$')
    ax.set_xlim([40000, 41000])
    ax.set_ylim([-0.8, 1.0])
    ax.set_xlabel("Sample index (-)")
    ax.set_ylabel("Normalized output (-)")
    ax.legend(loc="upper right")#bbox_to_anchor=(0.9, 1.3), frameon=True)
    plt.savefig("wh_best_timetrace.pdf")

    #%%
    plt.figure()
    plt.plot(model_data["TRAIN_LOSS"], 'b')
    plt.plot(model_data["VAL_LOSS"], 'r')

    #%% Metrics

    e_rms = 1000 * metrics.rmse(y, y_sim)[0]
    fit_idx = metrics.fit_index(y, y_sim)[0]
    r_sq = metrics.r_squared(y, y_sim)[0]

    print(f"RMSE: {e_rms:.1f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")
