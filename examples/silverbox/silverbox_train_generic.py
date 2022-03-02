import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
from torchid.ss.dt.models import NeuralStateUpdate, LinearOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator, FeedForwardStateEstimator
from loader import silverbox_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Parameters
    n_fit = 40000
    subseq_len = 256
    subseq_est_len = 20
    backward_est = True  # run the estimator backward in time
    batch_size = 64
    lr = 1e-3
    epochs = 10
    n_x = 2
    n_u = 1
    n_y = 1
    d_max = 3

    # Load dataset
    t_train, u_train, y_train = silverbox_loader("train", scale=True)

    #%% Prepare dataset
    load_len = subseq_len if backward_est else subseq_len + subseq_est_len
    train_data = SubsequenceDataset(u_train, y_train, subseq_len=load_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    f_xu = NeuralStateUpdate(n_x, n_u, d_max)
    g_x = LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    #state_estimator = LSTMStateEstimator(n_u=1, n_y=1, n_x=2, flipped=backward_est)
    state_estimator = FeedForwardStateEstimator(n_u=1, n_y=1, n_x=2, seq_len=subseq_est_len)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': state_estimator.parameters(), 'lr': lr},
    ], lr=lr)

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []

    start_time = time.time()

    # Training loop
    itr = 0
    for epoch in range(epochs):
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_est_u = batch_u[:subseq_est_len]
            batch_est_y = batch_y[:subseq_est_len]
            batch_x0 = state_estimator(batch_est_u, batch_est_y)

            if backward_est:
                batch_u_fit = batch_u
                batch_y_fit = batch_y
            else:
                batch_u_fit = batch_u[subseq_est_len:]
                batch_y_fit = batch_y[subseq_est_len:]

            batch_y_sim = model(batch_x0, batch_u_fit)
            loss = torch.nn.functional.mse_loss(batch_y, batch_y_sim)

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f'Iteration {itr} | AE Loss {loss:.4f} ')
            itr += 1

        print(f'Epoch {epoch} | AE Loss {loss:.4f} ')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%% Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = "ss_poly.pt"
    torch.save({"n_x": n_x,
                "n_y": n_y,
                "n_u": n_u,
                "d_max": d_max,
                "model": model.state_dict(),
                "estimator": state_estimator.state_dict()
                },
               os.path.join("models", model_filename))

    #%% Simulate
    t_full, u_full, y_full = silverbox_loader("full", scale=True)
    with torch.no_grad():
        u_v = torch.tensor(u_full[:, None, :])
        y_v = torch.tensor(y_full[:, None, :])
        x0 = state_estimator(u_v, y_v)
        y_sim = model(x0, u_v).squeeze(1)

    #%% Test
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_full[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label='sim')
