import os
import numpy as np
import scipy.io as sio
import matplotlib
#matplotlib.use("TKAgg")

data_folder = "Robot_Identification_Benchmark_Without_Raw_Data"
data_file = "forward_identification_without_raw_data.mat"


def robot_loader(dataset, scale=False, dtype=np.float32):

    data = sio.loadmat(os.path.join("data", data_folder, data_file))

    t = data[f"time_{dataset}"].ravel()
    u = data[f"u_{dataset}"].transpose()
    q = np.pi/180.0*data[f"y_{dataset}"].transpose()
    ts = 0.1  # np.median(np.diff(t))
    nq = q.shape[1]

    # Banal velocity computation
    v = np.r_[np.zeros((1, nq)), np.diff(q, axis=0)/ts]
    x = np.c_[q, v]

    #u = u - u[0, :]
    if scale:
        u_mean, u_std = robot_scaling()
        #u = u/u_std#
        u = (u - u_mean)/u_std
    return t.astype(dtype), u.astype(dtype), q.astype(dtype), v.astype(dtype)


def robot_scaling():
    data = sio.loadmat(os.path.join("data", data_folder, data_file))
    u_train = data["u_train"].transpose()
    #u_train = u_train - u_train[0, :]
    u_mean, u_std = np.mean(u_train, axis=0), np.std(u_train, axis=0)
    return u_mean, u_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for dataset in ["test", "train"]:
        t, u, q, v = robot_loader(dataset, scale=False)

        ts = 0.1
        nq = 6

        n_dof = q.shape[-1]
        a = np.r_[np.zeros((1, nq)), np.diff(v, axis=0) / ts]

        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(t, u)
        ax[1].plot(t, q)
        ax[2].plot(t, v)
        ax[3].plot(t, a)
        plt.suptitle(dataset)

    plt.figure()
    plt.plot(t, q[:, 0])
    plt.plot(t, v[:, 0])
    plt.plot(t, np.zeros(v[:, 0].shape), 'k')
