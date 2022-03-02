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
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Koopman spectrum estimation')
    parser.add_argument('--experiment_id', type=int, default=-1, metavar='N',
                        help='experiment id (default: -1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default:64)')
    parser.add_argument('--seq_len', type=int, default=80, metavar='N',
                        help='length of the training sequences (default: 20000)')
    parser.add_argument('--est_frac', type=float, default=0.63, metavar='N',
                        help='fraction of the subsequence used for initial state estimation')
    parser.add_argument('--est_direction', type=str, default="backward",
                        help='Estimate forward in time')
    parser.add_argument('--est_type', type=str, default="LSTM",
                        help='Estimator type')
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
    parser.add_argument('--no-figures', action='store_true', default=False,
                        help='Plot figures')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # CPU/GPU required resources
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.set_num_threads(args.n_threads)

    # Parameters
    seq_est_len = int(args.seq_len * args.est_frac)
    n_x = 6
    n_u = 1
    n_y = 1

    backward_est = True if args.est_direction == "backward" else False

    # Load dataset
    t_train, u_train, y_train = wh2009_loader("train", scale=True)

    #%%  Prepare dataset
    load_len = args.seq_len if backward_est else args.seq_len + seq_est_len
    train_data = SubsequenceDataset(u_train, y_train, subseq_len=load_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    f_xu = models.NeuralLinStateUpdate(n_x, n_u, n_feat=args.hidden_size).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    state_estimator = LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                         hidden_size=args.est_hidden_size,
                                         flipped=backward_est).to(device)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': state_estimator.parameters(), 'lr': args.lr},
    ], lr=args.lr)

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []

    # Training loop
    itr = 0
    for epoch in range(args.epochs):
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first

            batch_est_u = batch_u[:seq_est_len]
            batch_est_y = batch_u[:seq_est_len]
            batch_x0 = state_estimator(batch_est_u, batch_est_y)

            batch_est_u = batch_u[:seq_est_len]
            batch_est_y = batch_u[:seq_est_len]
            batch_x0 = state_estimator(batch_est_u, batch_est_y)

            if backward_est:
                # fit on the whole dataset
                batch_u_fit = batch_u
                batch_y_fit = batch_y
            else:
                # fit only after seq_est_len
                batch_u_fit = batch_u[seq_est_len:]
                batch_y_fit = batch_y[seq_est_len:]

            batch_y_sim = model(batch_x0, batch_u_fit)

            # Compute fit loss
            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f'Iteration {itr} | AE Loss {loss:.4f} ')
                if args.dry_run:
                    break
            itr += 1

        print(f'Epoch {epoch} | AE Loss {loss:.4f} ')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%% Save model
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.experiment_id >= 0:
        filename = f"model_{args.experiment_id}.pt"
    else:
        filename = "model.pt"

    torch.save({"n_x": n_x,
                "n_y": n_y,
                "n_u": n_u,
                "model": model.state_dict(),
                "estimator": state_estimator.state_dict()
                },
               os.path.join(args.save_folder, filename))

    #%% Simulate
    t_full, u_full, y_full = wh2009_loader("full", scale=True)
    with torch.no_grad():
        u_v = torch.tensor(u_full[:, None, :])
        y_v = torch.tensor(y_full[:, None, :])
        x0 = state_estimator(u_v, y_v)
        y_sim = model(x0, u_v).squeeze(1).detach().numpy()

    #%% Metrics
    from torchid import metrics
    e_rms = 1000 * metrics.rmse(y_full, y_sim)[0]
    fit_idx = metrics.fit_index(y_full, y_sim)[0]
    r_sq = metrics.r_squared(y_full, y_sim)[0]

    print(f"RMSE: {e_rms:.1f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

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

