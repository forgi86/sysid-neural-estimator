import os
import torch
import numpy as np
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
from loader import robot_loader, robot_scaling
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    model_data = torch.load(os.path.join("models", "model.pt"))

    n_dof = 6
    n_x = 2*n_dof
    n_u = n_dof
    n_y = n_x
    ts = 0.1

    args = model_data["args"]

    # Load dataset
    t, u, y = robot_loader("train", scale=True)
    u_mean, u_std = robot_scaling()

    #dataset = SubsequenceDataset(torch.tensor(u), torch.tensor(y), subseq_len=3000)
    dataset = SubsequenceDataset(torch.tensor(u), torch.tensor(y), subseq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    #%% Load models and parameters
    f_xu = models.MechanicalTrigStateSpaceSystem(n_dof, ts=ts, hidden_size=args.hidden_size)
    g_x = None
    model = StateSpaceSimulator(f_xu, g_x)
    model.load_state_dict(model_data["model"])

    val_loss = 0.0
    with torch.no_grad():
        for batch_u, batch_y in loader:

            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first
            batch_x0 = batch_y[0, :, :].squeeze(0)

            batch_y_sim = model(batch_x0, batch_u)

            # Compute fit loss
            loss = torch.nn.functional.mse_loss(batch_y, batch_y_sim)
            val_loss += loss

    val_loss = val_loss/len(loader)

    print(f"Val loss: {val_loss:.3f}")
    #%%
    batch_y_sim = batch_y_sim.numpy()
    batch_y = batch_y.numpy()


    examples = 4
    #fig, ax = plt.subplots(examples, 1, sharex=True)
    for batch_idx in range(examples):
        fig, ax = plt.subplots(3, 6, sharex=True)
        for dof_idx in range(n_dof):
            ax[0, dof_idx].plot(batch_y[:, batch_idx, dof_idx], 'k')
            ax[0, dof_idx].plot(batch_y_sim[:, batch_idx, dof_idx], 'b')
            ax[0, dof_idx].plot(batch_y[:, batch_idx, dof_idx] -
                                batch_y_sim[:, batch_idx, dof_idx], 'r')
            ax[0, dof_idx].set_ylim([-np.pi, np.pi])

            ax[1, dof_idx].plot(batch_y[:, batch_idx, dof_idx + n_dof], 'k')
            ax[1, dof_idx].plot(batch_y_sim[:, batch_idx, dof_idx + n_dof], 'b')
            ax[1, dof_idx].plot(batch_y[:, batch_idx, dof_idx + n_dof] -
                                batch_y_sim[:, batch_idx, dof_idx+n_dof], 'r')
            ax[2, dof_idx].plot(batch_u[:, batch_idx, dof_idx])
