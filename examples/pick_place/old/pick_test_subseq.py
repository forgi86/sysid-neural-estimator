import os
import torch
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
from loader import pick_place_loader, pick_place_scaling
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from torchid import metrics


if __name__ == '__main__':

    # model_data = torch.load(os.path.join("models", "model.pt"))

    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    args = model_data["args"]

    # Derived parameters
    if "est_frac" in args and args.est_frac is not None:
        seq_est_len = int(args.seq_len * args.est_frac)

    if "seq_est_len" in args and args.seq_est_len is not None:
        seq_est_len = args.seq_est_len
    backward_est = True if args.est_direction == "backward" else False
    if backward_est:
        load_len = max(args.seq_len, seq_est_len)
    else:
        load_len = args.seq_len + seq_est_len

    estimate_state = True #False

    # Load dataset
    t, u, y = wh2009_loader("test", scale=True)
    y_mean, y_std = wh2009_scaling()

    dataset = SubsequenceDataset(u, y, subseq_len=load_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    u_t = torch.tensor(u[:, None, :])
    y_t = torch.tensor(y[:, None, :])

    #%% Load models and parameters
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=args.hidden_size)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size)  #LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    if args.est_type == "LSTM":
        estimator = estimators.LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                  hidden_size=args.est_hidden_size,
                                                  flipped=backward_est)
    elif args.est_type == "FF":
        estimator = estimators.FeedForwardStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                         hidden_size=args.est_hidden_size,
                                                         seq_len=seq_est_len)
    elif args.est_type == "ZERO":
        estimator = estimators.ZeroStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x)
    else:
        raise ValueError("Wrong estimator type. Possible values: LSTM|FF|ZERO")

    model.load_state_dict(model_data["model"])
    estimator.load_state_dict(model_data["estimator"])

    val_loss = 0.0
    with torch.no_grad():
        for batch_u, batch_y in loader:
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_est_u = batch_u[:seq_est_len]
            batch_est_y = batch_y[:seq_est_len]
            if estimate_state:
                batch_x0 = estimator(batch_est_u, batch_est_y)
            else:
                batch_x0 = torch.zeros((batch_est_y.shape[1], n_x),
                                       dtype=batch_est_y.dtype,
                                       device=batch_est_y.device)

            if backward_est:
                # fit on the whole dataset
                batch_u_fit = batch_u[:args.seq_len]
                batch_y_fit = batch_y[:args.seq_len]
            else:
                # fit only after seq_est_len
                batch_u_fit = batch_u[seq_est_len:seq_est_len+args.seq_len]
                batch_y_fit = batch_y[seq_est_len:seq_est_len++args.seq_len]

            batch_y_sim = model(batch_x0, batch_u_fit)

            # Compute fit loss
            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)
            val_loss += loss

    val_loss = val_loss/len(loader)

    print(f"Val loss: {val_loss:.3f}")
    #%%
    batch_y_sim_np = batch_y_sim.squeeze(-1).transpose(0, 1).numpy()
    batch_y_fit_np = batch_y_fit.squeeze(-1).transpose(0, 1).numpy()

    examples = 4
    fig, ax = plt.subplots(examples, 1, sharex=True)
    for idx in range(examples):
        ax[idx].plot(batch_y_sim_np[idx], 'k')
        ax[idx].plot(batch_y_fit_np[idx], 'b')
        ax[idx].plot(batch_y_fit_np[idx] - batch_y_sim_np[idx], 'r')