import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator
from loader import wh2009_loader
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

if __name__ == '__main__':

    start_time = time.time()

    # Parameters
    subseq_len = 80  # 512 # 80 in Beintema (2020)
    subseq_est_len = 50
    batch_size = 1024  # 64
    lr = 1e-3
    epochs = 150
    hidden_size = 15
    n_fit = 80000
    n_x = 6
    n_u = 1
    n_y = 1

    # Load dataset
    t_train, u_train, y_train = wh2009_loader("train", scale=True)
    t_fit, u_fit, y_fit = t_train[:n_fit], u_train[:n_fit], y_train[:n_fit]
    t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

    # %% Prepare dataset
    train_data = SubsequenceDataset(u_fit, y_fit, subseq_len=subseq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    u_val_t = torch.tensor(u_val[:, None, :])
    y_val_t = torch.tensor(y_val[:, None, :])

    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=hidden_size)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=hidden_size)  # LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    state_estimator = LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x, hidden_size=32, flipped=True)

    # %% Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': state_estimator.parameters(), 'lr': lr},
    ], lr=lr)

    # %% Other initializations
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = "ss_nn.pt"

    ITR_LOSS, VAL_LOSS, TRAIN_LOSS = [], [], []
    min_loss = torch.inf

    # %% Training loop
    itr = 0
    for epoch in range(epochs):
        train_loss = 0  # train loss for the whole epoch
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_x0 = state_estimator(batch_u[:subseq_est_len], batch_y[:subseq_est_len])
            batch_y_sim = model(batch_x0, batch_u)

            # Compute autoencoder loss
            loss = torch.nn.functional.mse_loss(batch_y, batch_y_sim)
            train_loss += loss.item()

            # Statistics
            # ITR_LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f'Iteration {itr} | iteration loss {loss:.4f} ')
            itr += 1

        train_loss = train_loss/len(train_loader)
        TRAIN_LOSS.append(train_loss)

        # Validation loss: full simulation error
        with torch.no_grad():
            x0 = torch.zeros((1, n_x), dtype=u_val_t.dtype,
                             device=u_val_t.device)
            # x0 = state_estimator(u_val_t, y_val_t)
            y_val_sim = model(x0, u_val_t)
            val_loss = torch.nn.functional.mse_loss(y_val_t, y_val_sim)

        # best model so far, save it
        if val_loss < min_loss:
            torch.save({
                        "epoch": epoch,
                        "hidden_size": hidden_size,
                        "n_x": n_x,
                        "n_y": n_y,
                        "n_u": n_u,
                        "hidden_size": hidden_size,
                        "model": model.state_dict(),
                        "estimator": state_estimator.state_dict()
                        },
                       os.path.join("models", model_filename)
                       )
        VAL_LOSS.append(val_loss.item())
        print(f'==== Epoch {epoch} | Val Loss {val_loss:.4f} ====')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # %% Simulate
    # t_full, u_full, y_full = wh2009_loader("full", scale=True)
    # with torch.no_grad():
    #     u_v = torch.tensor(u_full[:, None, :])
    #     y_v = torch.tensor(y_full[:, None, :])
    #     x0 = state_estimator(u_v, y_v)
    #     y_full_sim = model(x0, u_v).squeeze(1).detach().numpy()

    # %% Metrics.

    model_data = torch.load(os.path.join("models", model_filename))
    model = StateSpaceSimulator(f_xu, g_x)
    state_estimator = LSTMStateEstimator(n_u=n_u, n_y=n_y, n_x=n_x, flipped=True)

    from torchid import metrics

    y_val_sim = y_val_sim.squeeze(1).detach().numpy()
    e_rms = 1000 * metrics.rmse(y_val, y_val_sim)[0]
    fit_idx = metrics.fit_index(y_val, y_val_sim)[0]
    r_sq = metrics.r_squared(y_val, y_val_sim)[0]

    print(f"RMSE: {e_rms:.1f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

    # %% Test
    fig, ax = plt.subplots(1, 1)
    ax.plot(TRAIN_LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_val[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_val_sim[:, 0], 'b', label='sim')
