import os
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt

data_folder = "Robot_Identification_Benchmark_Without_Raw_Data"
data_file = "forward_identification_without_raw_data.mat"

if __name__ == "__main__":
    data = sio.loadmat(os.path.join("data", data_folder, data_file))
    t = data["time_train"].ravel()
    u = data["u_train"].transpose()
    y = data["y_train"].transpose()

    ts = np.median(np.diff(t))
    ny = y.shape[1]
    nx = 2*ny

    # Banal velocity computation
    v = np.r_[np.zeros((1, ny)), np.diff(y, axis=0)/ts]
    fig, ax = plt.subplots()
    plt.figure()
    plt.plot(t, u)

    fig, ax = plt.subplots(3, ny, sharex=True)
    for i in range(ny):
        ax[0, i].plot(t, y[:, i])
        ax[1, i].plot(t, v[:, i])
        ax[2, i].plot(t, u[:, i])
