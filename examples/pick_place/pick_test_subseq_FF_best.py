import os
import numpy as np
import torch
import sys
sys.path.append("G:\My Drive\ID SUB SEQ\ident-subseq")

import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset


from loader import pick_place_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #model_data = torch.load(os.path.join("models", "model.pt"), map_location=torch.device('cpu'))
    #model_data = torch.load(os.path.join("models", "doe1", "model_67.pt"), map_location=torch.device('cpu'))
    #model_data = torch.load(os.path.join("models", "doe1", "model_55.pt"), map_location=torch.device('cpu')) # (10,64)
    model_data = torch.load(os.path.join("models", "doe1", "model_297.pt"), map_location=torch.device('cpu')) # (40+256)

    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    args = model_data["args"]
    decimate = 10

    t_full, u_full, y_full = pick_place_loader(dataset="full", decimate=decimate, scale=True)
    
    dataset = SubsequenceDataset(u_full, y_full, subseq_len=args.seq_len + args.seq_est_len)
    u_t = torch.tensor(u_full[:, None, :])
    y_t = torch.tensor(y_full[:, None, :])
    
    #%%
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
    
    val_loss = 0.0
    g = torch.manual_seed(1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    seq_est_len = args.seq_est_len
    with torch.no_grad():
        for batch_u, batch_y in loader:

            #a
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_est_u = batch_u[:seq_est_len]
            batch_est_y = batch_y[:seq_est_len]
            batch_x0 = estimator(batch_est_u, batch_est_y)

            batch_u_est = batch_u[:args.seq_est_len]
            batch_y_est = batch_y[:args.seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            batch_y_fit = batch_y[args.seq_est_len:]

            if args.est_type not in ["ZERO", "RAND"]:  # for not-dummy estimators
                batch_u_fit = batch_u[args.seq_est_len:]
            else:
                batch_u_fit = batch_u

            batch_y_sim = model(batch_x0, batch_u_fit)

            batch_y_sim_full = batch_y_sim.clone()
            if args.est_type not in ["ZERO", "RAND"]:
                batch_y_sim_full = torch.cat((torch.Tensor([float('NaN')])*torch.zeros((args.seq_est_len, args.batch_size, 1)), batch_y_sim_full), 0)

            # Compute fit loss
            # Compute fit loss
            if args.est_type in ["ZERO", "RAND"]:  # for dummy estimators
                batch_y_sim = batch_y_sim[args.seq_est_len:]

            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)
            val_loss += loss.item()

    val_loss = val_loss/len(loader)

    print(f"Val loss: {val_loss:.3f}")
    #%%
    batch_y_sim_full_np = batch_y_sim_full.squeeze(-1).transpose(0, 1).numpy()
    batch_y_np = batch_y.squeeze(-1).transpose(0, 1).numpy()
    
    
    batch_y_sim_full_np_worst = np.load("subseq_pp_zero.npy")

    examples = 3
    fig, ax = plt.subplots(examples, 1, sharex=True, figsize=(8, 6))
    #plt.suptitle("Training sequences")
    for idx in range(examples):
        ax[idx].plot(batch_y_np[idx], 'k', label="Training sequence")
        ax[idx].plot(batch_y_sim_full_np[idx], 'b--', label="$\hat{y}$ FF estimator")
        ax[idx].plot(batch_y_sim_full_np_worst[idx], 'm--', label=" $\hat{y}$ ZERO estimator")
        ax[idx].tick_params(axis='both', which='major', labelsize=16)
        #ax[idx].set_ylim((-2, 3))

        if idx == 0:
            ax[idx].legend(loc="upper right", fontsize = 16)
        #ax[idx].plot(batch_y_np[idx] - batch_y_sim_full_np[idx], 'r-.')
        if idx == examples - 1:
            ax[idx].set_xlabel("Training (estimation + fitting) sequence index (-)", fontsize = 16)
        ax[idx].grid(axis="x")
        
    plt.savefig("pp_subseq_best_worst.pdf")
