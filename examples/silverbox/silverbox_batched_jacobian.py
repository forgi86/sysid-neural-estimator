import os
import torch
from torchid.ss.dt.models import PolynomialStateUpdate, LinearStateUpdate, LinearOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator
from torch.utils.data import DataLoader
from loader import silverbox_loader
from torchid.datasets import SubsequenceDataset


if __name__ == '__main__':

    model_filename = "ss_poly.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    d_max = model_data["d_max"]

    # Load dataset
    t, u, y = silverbox_loader("test", scale=True)

    #%% Load models and parameters
    f_xu = PolynomialStateUpdate(n_x, n_u, d_max)
    g_x = LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    state_estimator = LSTMStateEstimator(n_u=n_u, n_y=n_y, n_x=n_x)
    model.load_state_dict(model_data["model"])
    #state_estimator.load_state_dict(model_data["estimator"])

    train_data = SubsequenceDataset(u, y, subseq_len=100)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    batch_u, batch_y = next(iter(train_loader))

    basis_x = torch.eye(n_x).unbind()
    n_x = 2
    dim_time = 1
    x = []
    x_0 = torch.zeros(64, n_x)
    n_param = sum(map(torch.numel, model.parameters()))
    s_step = torch.zeros(n_x, n_param)
    x_step = x_0

    var = torch.zeros_like(x_0)
    vars = []
    for idx in range(n_x):
        var_i = var.clone()
        var_i[..., idx] = 1.0
        vars.append(var_i)


    for u_step in batch_u.split(1, dim=dim_time):  # split along the time axis
        u_step = u_step.squeeze(dim_time)
        x += [x_step]
        x_step_ = x_step.detach().clone().requires_grad_(True)
        delta_x = model.f_xu(x_step_, u_step)
        x_step = x_step + delta_x

        # Jacobian wrt x
        J_x = [torch.autograd.grad(delta_x, x_step_, v, retain_graph=True)[0] for v in vars]
        J_x = torch.stack(J_x, dim=-2)

        # Jacobian wrt theta
        jacs_theta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in vars]
        jacs_theta_f = [torch.cat([jac.ravel() for jac in jacs_theta[j]]) for j in range(n_x)]  # ravel jacobian rows
        J_theta = torch.stack(jacs_theta_f)  # stack jacobian rows to obtain a jacobian matrix

    x = torch.stack(x, dim_time)
