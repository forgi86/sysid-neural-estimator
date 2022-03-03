import numpy as np
import torch
import time
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = torch.device("gpu")

    n_x = 6
    n_u = 1
    n_y = 1
    batch_size = 64
    seq_len = 64
    seq_est_len = seq_len  # 16

    #%% Load models and parameters
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=16).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=16).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    estimator = estimators.LSTMStateEstimator(n_u=n_y, n_y=n_y,
                                              n_x=n_x, hidden_size=16,
                                              flipped=True).to(device)

    SEQ_LEN = [16, 32, 64, 128, 256, 512, 1024]
    BATCH_SIZE = [1, 32, 64, 128, 512, 1024]
    TIME = np.empty((len(SEQ_LEN), len(BATCH_SIZE)))
    PROCESSED_SAMPLES = np.empty((len(SEQ_LEN), len(BATCH_SIZE)))
    for i, seq_len in enumerate(SEQ_LEN):
        for j, batch_size in enumerate(BATCH_SIZE):
            PROCESSED_SAMPLES[i, j] = seq_len * batch_size
            batch_u = torch.randn(seq_len, batch_size, n_u).to(device)
            batch_y = torch.randn(seq_len, batch_size, n_y).to(device)

            time_start = time.time()
            for rep in range(10):

                model.zero_grad()
                estimator.zero_grad()
                batch_est_u = batch_u[:seq_est_len]
                batch_est_y = batch_y[:seq_est_len]
                batch_x0 = estimator(batch_est_u, batch_est_y)

                batch_u_fit = batch_u
                batch_y_fit = batch_y

                batch_y_sim = model(batch_x0, batch_u_fit)
                # Compute fit loss
                loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)

                loss.backward()

            time_proc = time.time() - time_start
            TIME[i, j] = time_proc
            print(i, j, time_proc)

    SAMPLES_SEC = PROCESSED_SAMPLES/TIME

    #%% Fixed sequence length, increasing batch_size
    plt.figure()
    plt.plot(SEQ_LEN, TIME)
    plt.xlabel("Sequence length")
    plt.legend(BATCH_SIZE)
    plt.ylabel("Time (s)")

    plt.figure()
    plt.plot(SEQ_LEN, SAMPLES_SEC)
    plt.xlabel("Sequence length")
    plt.legend(BATCH_SIZE)
    plt.ylabel("Samples/Time (1/s)")

    #%% Fixed batch size, increasing sequence length
    plt.figure()
    plt.plot(BATCH_SIZE, TIME.transpose())
    plt.xlabel("Batch size")
    plt.legend(SEQ_LEN)
    plt.ylabel("Time (s)")

    plt.figure()
    plt.plot(BATCH_SIZE, SAMPLES_SEC.transpose())
    plt.xlabel("Batch size")
    plt.legend(SEQ_LEN)
    plt.ylabel("Samples/Time (1/s)")

