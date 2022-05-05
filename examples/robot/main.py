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

    # example parameters:
    # --lr 0.001 --epochs 10000 --max_time 60 --batch_size 1024 --seq_len 80
    # --est_frac 0.63 --est_direction forward --est_type FF --est_hidden_size 32
    parser = argparse.ArgumentParser(description='State-space neural network tests')
    parser.add_argument('--experiment_id', type=int, default=-1, metavar='N',
                        help='experiment id (default: -1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--max_time', type=float, default=6*3600, metavar='N',
                        help='maximum training time in seconds (default:3600)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default:64)')
    parser.add_argument('--seq_len', type=int, default=70, metavar='N',
                        help='length of the training sequences (default: 20000)')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                        help='estimator: number of units per hidden layer (default: 64)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
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
    t_train, u_train, y_train = robot_loader("train", scale=True)
    t_fit, u_fit, y_fit = t_train[:n_fit], u_train[:n_fit], y_train[:n_fit]
    t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

    # %%  Prepare dataset, models, optimizer
    train_data = SubsequenceDataset(u_fit, y_fit, subseq_len=args.seq_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = SubsequenceDataset(torch.tensor(u_val), torch.tensor(y_val), subseq_len=args.seq_len)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    #f_xu = models.MechanicalTrigStateSpaceSystem(n_dof=n_dof, ts=ts, hidden_size=args.hidden_size, init_small=False).to(device)
    f_xu = models.MechanicalTrigStateSpaceSystem(n_dof=n_dof, ts=ts, hidden_size=args.hidden_size, init_small=True).to(device)

    #f_xu = models.MechanicalTrigStateSpaceSystemV2(n_dof=n_dof, ts=ts, init_small=False).to(device)

    model = StateSpaceSimulator(f_xu, g_x=None).to(device)

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
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            if (time.time() - start_time) >= args.max_time:
                break

            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first
            batch_x0 = batch_y[0, :, :].squeeze(0)
            batch_y_sim = model(batch_x0, batch_u)

            #loss = torch.nn.functional.mse_loss(batch_y, batch_y_sim)
            loss = torch.nn.functional.mse_loss(batch_y[..., :n_dof], batch_y_sim[..., :n_dof])
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
            for batch_u, batch_y in val_loader:
                batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
                batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first
                batch_x0 = batch_y[0, :, :].squeeze(0)

                batch_y_sim = model(batch_x0, batch_u)

                # Compute val loss
                #loss = torch.nn.functional.mse_loss(batch_y, batch_y_sim)
                loss = torch.nn.functional.mse_loss(batch_y[..., :n_dof], batch_y_sim[..., :n_dof])
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
    torch.save(model_data, model_path)

    # Reload optimal parameters (best on validation)
    model.load_state_dict(model_data["model"])
    #estimator.load_state_dict(model_data["estimator"])

    t_test, u_test, y_test = robot_loader("test", scale=True)
    q_test = y_test[:, :n_dof]
    v_test = y_test[:, n_dof:]
    with torch.no_grad():
        model.eval()
        #estimator.eval()
        u_v = torch.tensor(u_test[:, None, :]).to(device)
        y_v = torch.tensor(y_test[:, None, :]).to(device)
        # x0 = estimator(u_v, y_v)
        x0 = torch.zeros((1, n_x), dtype=u_v.dtype, device=u_v.device)
        y_sim = model(x0, u_v).squeeze(1).to("cpu").detach().numpy()
        q_sim = y_sim[:, :n_dof]
        v_sim = y_sim[:, n_dof:]
    # %% Metrics

    from torchid import metrics
    e_rms = metrics.rmse(y_test, y_sim)
    fit_idx = metrics.fit_index(y_test, y_sim)
    r_sq = metrics.r_squared(y_test, y_sim)

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

        fig, ax = plt.subplots(2, 6, sharex=True)
        for idx in range(n_dof):
            ax[0, idx].plot(t_test, q_test[:, idx], 'k')
            ax[0, idx].plot(t_test, q_sim[:, idx], 'b')
            ax[0, idx].plot(t_test, q_test[:, idx] - q_sim[:, idx], 'r')
            ax[1, idx].plot(t_test, v_test[:, idx], 'k')
            ax[1, idx].plot(t_test, v_sim[:, idx], 'b')
            ax[1, idx].plot(t_test, v_test[:, idx] - v_sim[:, idx], 'r')
        plt.suptitle("Test")


