import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import wh2009_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    # example parameters:
    # --lr 0.001 --epochs 10000 --max_time 60 --batch_size 1024 --seq_len 80
    # --est_frac 0.63 --est_direction forward --est_type FF --est_hidden_size 32
    parser = argparse.ArgumentParser(description='State-space neural network tests')
    parser.add_argument('--experiment_id', type=int, default=-1, metavar='N',
                        help='experiment id (default: -1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--max_time', type=float, default=60, metavar='N',
                        help='maximum training time in seconds (default:3600)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default:64)')
    parser.add_argument('--seq_len', type=int, default=80, metavar='N',
                        help='length of the training sequences (default: 20000)')
    parser.add_argument('--seq_est_len', type=int, default=50, metavar='N',
                        help='length of the training sequences (default: 20000)')
    parser.add_argument('--est_frac', type=float, default=None, metavar='N',
                        help='fraction of the subsequence used for initial state estimation')
    parser.add_argument('--est_direction', type=str, default="forward",
                        help='Estimate forward in time')
    parser.add_argument('--est_type', type=str, default="LSTM",
                        help='Estimator type. Possible values: LSTM|FF|ZERO')
    parser.add_argument('--est_hidden_size', type=int, default=16, metavar='N',
                        help='model: number of units per hidden layer (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=15, metavar='N',
                        help='estimator: number of units per hidden layer (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--save-folder', type=str, default="models", metavar='S',
                        help='save folder (default: "model")')

    parser.add_argument('--n-threads', type=int, default=2,
                        help='number of CPU threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='log interval')
    parser.add_argument('--no-figures', action='store_true', default=False,
                        help='Plot figures')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Derived parameters
    if "est_frac" in args and args.est_frac is not None:
        seq_est_len = int(args.seq_len * args.est_frac)

    if "seq_est_len" in args and args.seq_est_len is not None:
        seq_est_len = args.seq_est_len

    if args.est_direction == "backward":
        backward_est = True
    elif args.est_direction == "forward":
        backward_est = False
    else:
        raise ValueError("Wrong estimator direction. Possible values: backward|forward")

    if backward_est:
        load_len = max(args.seq_len, seq_est_len)
    else:
        load_len = args.seq_len + seq_est_len

    # CPU/GPU resources
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(args.n_threads)

    # Constants
    n_x = 6
    n_u = 1
    n_y = 1
    n_fit = 80000

    # %% Load dataset
    t_train, u_train, y_train = wh2009_loader("train", scale=True)
    t_fit, u_fit, y_fit = t_train[:n_fit], u_train[:n_fit], y_train[:n_fit]
    t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

    # %%  Prepare dataset, models, optimizer
    train_data = SubsequenceDataset(u_train, y_train, subseq_len=load_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    u_val_t = torch.tensor(u_val[:, None, :]).to(device)
    y_val_t = torch.tensor(y_val[:, None, :]).to(device)

    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=args.hidden_size).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    if args.est_type == "LSTM":
        estimator = estimators.LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                  hidden_size=args.est_hidden_size,
                                                  flipped=backward_est)
    elif args.est_type == "FF":
        estimator = estimators.FeedForwardStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                         hidden_size=args.est_hidden_size,
                                                         seq_len=seq_est_len)
    elif args.est_type == "ZERO":
        estimator = estimators.ZeroStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x)
    else:
        raise ValueError("Wrong estimator type. Possible values: LSTM|FF|ZERO")

    estimator = estimator.to(device)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': estimator.parameters(), 'lr': args.lr},
    ], lr=args.lr)

    # %% Other initializations
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.experiment_id >= 0:
        model_filename = f"model_{args.experiment_id}.pt"
    else:
        model_filename = "model.pt"
    model_path = os.path.join(args.save_folder, model_filename)

    VAL_LOSS, TRAIN_LOSS = [], []
    min_loss = np.inf  # for early stopping

    start_time = time.time()
    # %% Training loop
    itr = 0
    for epoch in range(args.epochs):
        train_loss = 0  # train loss for the whole epoch
        model.train()
        estimator.train()
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            if (time.time() - start_time) >= args.max_time:
                break

            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first

            # State is estimated on the first seq_est_len samples
            batch_u_est = batch_u[:seq_est_len]
            batch_y_est = batch_y[:seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            if backward_est:
                # fit on the whole dataset
                batch_u_fit = batch_u[:args.seq_len]
                batch_y_fit = batch_y[:args.seq_len]
            else:
                # fit only after seq_est_len
                batch_u_fit = batch_u[seq_est_len:seq_est_len+args.seq_len]
                batch_y_fit = batch_y[seq_est_len:seq_est_len+args.seq_len]

            batch_y_sim = model(batch_x0, batch_u_fit)

            # Compute fit loss
            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)
            train_loss += loss.item()

            # ITR_LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if args.dry_run:
                break

            #            if itr % args.log_interval == 0:
            #                print(f'Iteration {itr} | iteration loss {loss:.4f} ')

            itr += 1

        train_loss = train_loss / len(train_loader)
        TRAIN_LOSS.append(train_loss)

        # Validation loss: full simulation error
        with torch.no_grad():
            model.eval()
            estimator.eval()
            x0 = torch.zeros((1, n_x), dtype=u_val_t.dtype,
                             device=u_val_t.device)
            # x0 = state_estimator(u_val_t, y_val_t)
            y_val_sim = model(x0, u_val_t)
            val_loss = torch.nn.functional.mse_loss(y_val_t, y_val_sim)

        VAL_LOSS.append(val_loss.item())
        print(f'==== Epoch {epoch} | Train Loss {train_loss:.4f} Val (sim) Loss {val_loss:.4f} ====')

        # best model so far, save it
        if val_loss < min_loss:
            torch.save({
                "epoch": epoch,
                "args": args,
                "time": time.time() - start_time,
                "n_x": n_x,
                "n_y": n_y,
                "n_u": n_u,
                "TRAIN_LOSS": TRAIN_LOSS,
                "VAL_LOSS": VAL_LOSS,
                "model": model.state_dict(),
                "estimator": estimator.state_dict()
            },
                os.path.join(model_path)
            )
            min_loss = val_loss.item()

        if args.dry_run:
            break

        if (time.time() - start_time) >= args.max_time:
            break

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    if not np.isfinite(min_loss):  # model never saved as it was never giving a finite simulation loss
        torch.save({
            "epoch": epoch,
            "args": args,
            "time": time.time() - start_time,
            "n_x": n_x,
            "n_y": n_y,
            "n_u": n_u,
            "TRAIN_LOSS": TRAIN_LOSS,
            "VAL_LOSS": VAL_LOSS,
            "model": model.state_dict(),
            "estimator": estimator.state_dict()
        },
            os.path.join(model_path)
        )
    # %% Simulate

    # Also save total training time (up to last epoch)
    model_data = torch.load(model_path)
    model_data["total_time"] = time.time() - start_time
    torch.save(model_data, model_path)

    # Reload optimal parameters (best on validation)
    model.load_state_dict(model_data["model"])
    estimator.load_state_dict(model_data["estimator"])

    t_full, u_full, y_full = wh2009_loader("full", scale=True)
    with torch.no_grad():
        model.eval()
        estimator.eval()
        u_v = torch.tensor(u_full[:, None, :]).to(device)
        y_v = torch.tensor(y_full[:, None, :]).to(device)
        # x0 = estimator(u_v, y_v)
        x0 = torch.zeros((1, n_x), dtype=u_v.dtype, device=u_v.device)
        y_sim = model(x0, u_v).squeeze(1).to("cpu").detach().numpy()

    # %% Metrics

    from torchid import metrics
    e_rms = 1000 * metrics.rmse(y_full, y_sim)[0]
    fit_idx = metrics.fit_index(y_full, y_sim)[0]
    r_sq = metrics.r_squared(y_full, y_sim)[0]

    print(f"RMSE: {e_rms:.1f} mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

    # %% Plots
    if not args.no_figures:
        fig, ax = plt.subplots(1, 1)
        ax.plot(TRAIN_LOSS, 'k', label='TRAIN')
        ax.plot(VAL_LOSS, 'r', label='VAL')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("Loss (-)")
        ax.set_xlabel("Iteration (-)")

        fig, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(y_full[:, 0], 'k', label='meas')
        ax.grid(True)
        ax.plot(y_sim[:, 0], 'b', label='sim')
