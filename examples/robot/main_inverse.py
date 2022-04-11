import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
import torchid.ss.dt.models as models
from loader import robot_loader
from torchid.ss.dt.simulator import StateSpaceSimulator
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import argparse
import time

if __name__ == "__main__":

    n_fit = 30000
    n_dof = 6
    ts = 0.1

    # %% Load dataset
    t_train, u_train, x_train = robot_loader("train", scale=True)
    q_train = x_train[:, :n_dof]
    v_train = x_train[:, n_dof:]

    a_train = np.r_[np.zeros((1, n_dof)), np.diff(v_train, axis=0)/ts]

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(t_train, u_train)
    ax[1].plot(t_train, q_train)
    ax[2].plot(t_train, v_train)
    ax[3].plot(t_train, a_train)
