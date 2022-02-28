import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape  # extra comma

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class LSTMStateEstimator(nn.Module):
    """ Black-box estimator from the sequences of (u, y) to x[N-1].
    The estimation is performed by processing (u, y) forward in time.
    """

    def __init__(self, n_u=1, n_y=1, n_x=2, hidden_size=16, batch_first=False, flipped=False):
        super(LSTMStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.batch_first = batch_first
        self.flipped = flipped

        self.lstm = nn.LSTM(input_size=n_y + n_u, hidden_size=hidden_size, batch_first=batch_first)
        self.lstm_output = nn.Linear(hidden_size, n_x)
        self.dim_time = 1 if self.batch_first else 0

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        if self.flipped:
            uy = uy.flip(self.dim_time)
        _, (hN, cN) = self.lstm(uy)
        xN = self.lstm_output(hN).squeeze(0)
        return xN


class LuenbergerStateEstimator(nn.Module):
    """ Black-box estimator from the sequences of (u, y) to x[N-1].
    The estimation is performed by processing (u, y) forward in time.
    """

    def __init__(self, f_xu, g_x=None, n_u=1, n_y=1, n_x=2, batch_first=False, flipped=False):
        super(LuenbergerStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.f_xu = f_xu
        self.g_x = g_x
        self.batch_first = batch_first
        self.flipped = flipped

        self.gain_net = nn.Sequential(
            nn.Linear(n_x, 64),
            nn.ReLU(),
            nn.Linear(64, n_x * n_y),
            View((n_x, n_y))
        )
        self.dim_time = 1 if self.batch_first else 0
        self.dim_batch = 0 if self.batch_first else 1

    def forward(self, u, y, x_0=None):
        x_hat = []
        if x_0 is None:
            x_0 = torch.zeros((u.shape[self.dim_batch], self.n_x), dtype=u.dtype, device=u.device)
        x_step = x_0

        if self.flipped:
            u = u.flip(self.dim_time)
            y = y.flip(self.dim_time)

        uy_seq = zip(u.split(1, dim=self.dim_time), y.split(1, self.dim_time))
        for u_step, y_step in uy_seq:  # split along the time axis
            y_hat = self.g_x(x_step)
            u_step = u_step.squeeze(self.dim_time)
            y_step = y_step.squeeze(self.dim_time)
            x_hat += [x_step]
            K_step = self.gain_net(x_step)
            e_step = y_hat - y_step
            dx = self.f_xu(x_step, u_step)
            x_step = (x_step + dx)  # predict
            x_step = x_step + K_step.matmul(e_step[:, :, None]).squeeze(-1)  # update

        # x_hat = torch.stack(x_hat, dim_time)
        return x_step


class FlippedLSTMStateEstimator(nn.Module):
    """ Black-box estimator from the sequences of (u, y) to x[0].
    The estimation is performed by processing (u, y) backward in time.
    """

    def __init__(self, n_u=1, n_y=1, n_x=2, hidden_size=16, batch_first=False):
        super(LSTMStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=n_y + n_u, hidden_size=hidden_size, batch_first=batch_first)
        self.lstm_output = nn.Linear(hidden_size, n_x)

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        uy_rev = uy.flip(0)
        _, (h0, c0) = self.lstm(uy_rev)
        x0 = self.lstm_output(h0).squeeze(0)
        return x0
