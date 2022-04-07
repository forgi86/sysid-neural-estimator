import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("TKAgg")

data_file = "headdata_1v6.mat"


def pick_place_loader(scale=True, dtype=np.float32):

    data = sio.loadmat(os.path.join("data", data_file))

    t = data["t"]
    u = data["u"]
    y = data["y"]

    if scale:
        u_mean, u_std, y_mean, y_std = pick_place_scaling()
        u = (u - u_mean)/u_std
        y = (y - y_mean)/y_std

    return t.astype(dtype), u.astype(dtype), y.astype(dtype)


def pick_place_scaling():
    data = sio.loadmat(os.path.join("data", data_file))
    u = data["u"]
    y = data["y"]
    u_mean, u_std = np.mean(u, axis=0), np.std(u, axis=0)
    y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
    return u_mean, u_std, y_mean, y_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, u, y = pick_place_loader(scale=True)
    ts = np.median(np.diff(t, axis=0))
    v = np.r_[[[0]], np.diff(y, axis=0)/ts]
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, u)
    ax[1].plot(t, y)
    ax[2].plot(t, v)
    plt.show()
