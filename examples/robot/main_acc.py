import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loader import robot_loader
import matplotlib.pyplot as plt
import argparse
import time
import tqdm


if __name__ == "__main__":

    # example parameters:
    # --lr 0.001 --epochs 10000 --max_time 60 --batch_size 1024 --seq_len 80
    # --est_frac 0.63 --est_direction forward --est_type FF --est_hidden_size 32
    parser = argparse.ArgumentParser(description='State-space neural network tests')
    parser.add_argument('--experiment_id', type=int, default=-1, metavar='N',
                        help='experiment id (default: -1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--max_time', type=float, default=1*3600, metavar='N',
                        help='maximum training time in seconds (default:3600)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default:64)')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                        help='estimator: number of units per hidden layer (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
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

    # CPU/GPU resources
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(args.n_threads)

    # Constants
    n_dof = 6
    n_x = 2*n_dof
    n_u = n_dof
    n_y = n_x
    n_fit = 30000
    ts = 0.1

    # %% Load dataset
    t_train, u_train, q_train, v_train = robot_loader("train", scale=True)
    a_train = np.r_[np.zeros((1, n_dof)), np.diff(v_train, axis=0) / ts].astype(np.float32)
    x_train = np.c_[q_train, v_train, u_train]
    #x_train = np.c_[np.cos(q_train), np.sin(q_train), v_train, u_train]

    t_fit, x_fit, a_fit = t_train[:n_fit], x_train[:n_fit], a_train[:n_fit]
    t_val, x_val, a_val = t_train[n_fit:], x_train[n_fit:], a_train[n_fit:]

    # %%  Prepare dataset, models, optimizer
    train_data = TensorDataset(torch.tensor(x_fit), torch.tensor(a_fit))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = TensorDataset(torch.tensor(x_val), torch.tensor(a_val))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    model = torch.nn.Sequential(
        nn.Linear(2 * n_dof + n_dof, 64),  # inputs: position, velocities, torques (fully actuated)
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_dof)
    )
    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
    ], lr=args.lr)

    # %% Other initializations
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.experiment_id >= 0:
        model_filename = f"model_{args.experiment_id}.pt"
    else:
        model_filename = "model_acc.pt"
    model_path = os.path.join(args.save_folder, model_filename)

    VAL_LOSS, TRAIN_LOSS = [], []
    min_loss = np.inf  # for early stopping

    start_time = time.time()
    # %% Training loop
    itr = 0
    for epoch in range(args.epochs):
        train_loss = 0  # train loss for the whole epoch
        model.train()
        for batch_idx, (batch_x, batch_a) in enumerate(train_loader):
            if (time.time() - start_time) >= args.max_time:
                break

            optimizer.zero_grad()

            # Compute fit loss
            batch_a_pred = model(batch_x)

            loss = torch.nn.functional.mse_loss(batch_a, batch_a_pred)
            train_loss += loss.item()

            # ITR_LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if args.dry_run:
                break

            itr += 1

        train_loss = train_loss / len(train_loader)
        TRAIN_LOSS.append(train_loss)

        # Validation loss: prediction on the same interval
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_x, batch_a in val_loader:

                batch_a_pred = model(batch_x)

                # Compute val loss
                loss = torch.nn.functional.mse_loss(batch_a, batch_a_pred)
                val_loss += loss

        val_loss = val_loss / len(val_loader)
        VAL_LOSS.append(val_loss.item())

        print(f'==== Epoch {epoch} | Train Loss {train_loss:.4f} Val Loss {val_loss:.4f} ====')

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
            #"estimator": estimator.state_dict()
        },
            os.path.join(model_path)
        )
    # %% Simulate

    # Also save total training time (up to last epoch)
    model_data = torch.load(model_path)
    model_data["total_time"] = time.time() - start_time
    model_data["TRAIN_LOSS"] = TRAIN_LOSS
    model_data["VAL_LOSS"] = VAL_LOSS
    torch.save(model_data, model_path)

    # Reload optimal parameters (best on validation)
    model.load_state_dict(model_data["model"])
    #estimator.load_state_dict(model_data["estimator"])

    t_test, u_test, q_test, v_test = robot_loader("test", scale=True)
    x_test = np.c_[q_test, v_test, u_test]
    #x_test = np.c_[np.cos(q_test), np.sin(q_test), v_test, u_test]
    a_test = np.r_[np.zeros((1, n_dof)), np.diff(v_test, axis=0) / ts].astype(np.float32)
    with torch.no_grad():
        model.eval()
        a_test_pred = model(torch.tensor(x_test)).detach().numpy()
    # %% Metrics

    from torchid import metrics
    e_rms = metrics.rmse(a_test, a_test_pred)
    fit_idx = metrics.fit_index(a_test, a_test_pred)
    r_sq = metrics.r_squared(a_test, a_test_pred)

    print("RMSE: " + str(e_rms))
    print("FIT: " + str(fit_idx))

    # %% Plots
    if not args.no_figures:
        fig, ax = plt.subplots(1, 1)
        ax.plot(TRAIN_LOSS, 'k', label='TRAIN')
        ax.plot(VAL_LOSS, 'r', label='VAL')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("Loss (-)")
        ax.set_xlabel("Iteration (-)")

        fig, ax = plt.subplots(1, 6, sharex=True, squeeze=False)
        for idx in range(n_dof):
            ax[0, idx].plot(t_test, a_test[:, idx], 'k')
            ax[0, idx].plot(t_test, a_test_pred[:, idx], 'b')
        plt.suptitle("Test")


