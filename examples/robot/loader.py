import os
import numpy as np
import scipy
import scipy.io as sio


data_folder = "Robot_Identification_Benchmark_Without_Raw_Data"
data_file = "forward_identification_without_raw_data.mat"


def robot_loader(dataset, scale=True, dtype=np.float32):

    data = sio.loadmat(os.path.join("data", data_folder, data_file))

    t = data[f"time_{dataset}"].ravel()
    u = data[f"u_{dataset}"].transpose()
    q = data[f"y_{dataset}"].transpose()*np.pi/180.0
    ts = np.median(np.diff(t))
    ny = q.shape[1]
    nx = 2*ny

    # Banal velocity computation
    dq = np.r_[np.zeros((1, ny)), np.diff(q, axis=0)/ts]

    if scale:
        scale_q, scale_v, scale_u = robot_scaling()
        q /= scale_q
        dq /= scale_v
        u /= scale_u
    return t, u, q, dq


def robot_scaling():
    scale_q = 1.0
    scale_v = 1.0
    scale_u = 1.0
    return scale_q, scale_v, scale_u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for dataset in ["train", "test"]:
        t, u, q, dq = robot_loader(dataset, scale=False)
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(t, u)
        ax[1].plot(t, q)
        ax[2].plot(t, dq)
        plt.suptitle(dataset)
