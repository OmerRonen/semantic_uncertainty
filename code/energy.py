import numpy as np
import torch

from tqdm import tqdm
from torch import nn
from torch.func import jacfwd
from torch.distributions import Distribution


def calculate_energy(X, model, mu=None, sigma=None, batch_size=32, fast=True):
    # calculate gaussian density of input
    model.eval()
    batch_size = min(batch_size, X.shape[0])
    # X = X.to(device="cpu", dtype=model.dtype)
    # mu = mu.to(device="cpu", dtype=model.dtype)
    if len(X.shape) == 3:
        # flatten the input except for the batch dimension
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    if mu is None:
        log_density = torch.zeros(X.shape[0])
    else:
        diff = (X - mu).unsqueeze(1).to(device="cpu", dtype=model.dtype)
        sigma = sigma.to(device="cpu", dtype=model.dtype)
        log_density = - 0.5 * (diff @ torch.inverse(sigma) @ diff.transpose(1, 2))

    n_batches = int(np.ceil(X.shape[0] / batch_size))
    det_grad_list = []
    for b in range(n_batches):
        X_b_gpu = X[b * batch_size: (b + 1) * batch_size, ...]  # .to(device=model.device, dtype=model.dtype)
        if fast:
            det_grad_list.append(calculate_det_grad_fast(X_b_gpu, model).detach().cpu())
        else:
            det_grad_list.append(calculate_det_grad(X_b_gpu, model).detach().cpu())
    det_grad = torch.cat(det_grad_list, dim=0)

    log_density = det_grad + torch.squeeze(log_density)
    return log_density


def get_energy_logits(pre_softmax):
    pre_softmax = pre_softmax.to(dtype=torch.float64)
    energy_total = None
    temp = 1
    if len(pre_softmax.shape) == 2:
        pre_softmax = pre_softmax.unsqueeze(0)
    vocab_size = pre_softmax.shape[2]
    energies = []
    for i in range(pre_softmax.shape[1]):
        token_pre_softmax = pre_softmax[:, i, :]
        token_pre_softmax = token_pre_softmax / temp
        probs = torch.softmax(token_pre_softmax, dim=-1)

        log_c = token_pre_softmax.logsumexp(dim=-1)
        one_over_probs = 1 / probs  # torch.clamp(1 / probs, min=1e-10, max=1e10)
        factor_mul_log = 2 * torch.sum(torch.log(one_over_probs))
        c_term = torch.exp(-2 * log_c)
        energy = factor_mul_log + torch.log(1 + torch.sum((probs ** 2)) * c_term)
        energies.append(energy / vocab_size)

        if energy_total is None:
            energy_total = energy / vocab_size
        else:
            energy_total += energy / vocab_size
    print(energies)
    return energy_total


def get_probs(net, X, pre_softmax=None, derivative=False, fast=False):
    with torch.no_grad():
        pre_softmax = net(X).detach().cpu() if pre_softmax is None else pre_softmax
        # LOGGER.debug(f"max {pre_softmax.max()} min {pre_softmax.min()}")

    is_binary = pre_softmax.shape[-1] == 2

    if is_binary and not derivative:
        ps = torch.clamp(pre_softmax[..., 1], min=-60, max=60)
        mx = pre_softmax[..., 1].max()
        mn = pre_softmax[..., 1].min()
        pre_softmax = 10 * ps / (mx - mn)

        probs = torch.sigmoid(pre_softmax)
        probs = 1 / (probs * (1 - probs))
        return probs.unsqueeze(2).transpose(1, 2)
    # pre_softmax = torch.clamp(pre_softmax, min=-30, max=30).detach().cpu()

    n_preds = pre_softmax.shape[0]
    seq_length = pre_softmax.shape[1]
    dict_size = pre_softmax.shape[2]
    out_size = seq_length * dict_size
    n_params = seq_length * (dict_size + 1)
    if not fast:
        probs_arr = torch.zeros(pre_softmax.shape[0], n_params, out_size)
    else:
        dets = None
    # probs_arr = None
    temp = 1
    for i in range(pre_softmax.shape[1]):
        token_pre_softmax = pre_softmax[:, i, :]
        token_pre_softmax = token_pre_softmax / temp
        probs = torch.softmax(token_pre_softmax, dim=1)

        log_c = token_pre_softmax.logsumexp(dim=1)
        c = torch.exp(log_c)
        c_vec = (c.view(-1, 1) * torch.ones(n_preds, dict_size)).unsqueeze(1)
        one_over_probs = 1 / probs  # torch.clamp(1 / probs, min=1e-10, max=1e10)
        if not fast:
            p_diag = torch.diag_embed(one_over_probs)

            mat_i = torch.cat([p_diag, c_vec], dim=1)

            probs_arr[:, i * (dict_size + 1): (i + 1) * (dict_size + 1),
            i * dict_size: (i + 1) * dict_size] = mat_i
        else:
            factor_mul_log = 2 * torch.sum(torch.log(one_over_probs))
            det = factor_mul_log + torch.log(torch.sum((probs ** 2))) + -2 * log_c
            # mat_rand = torch.randn(size=(X.shape[0], X.shape[1], mat_i.shape[1]))
            # mat_i = mat_rand @ mat_i
            # mat_i = mat_i @ mat_i.transpose(1, 2)
            # move mat_i to cpu numpy
            # mat_i = mat_i.detach().cpu()
            if dets is None:
                dets = det
            else:
                dets += det

    if not fast:

        return probs_arr
    else:
        return dets

    # return probs


def calculate_det_grad_fast(X, net):
    dets = get_probs(net, X, fast=True)  # [n_preds, seq_length, dict_size]
    return dets.unsqueeze(0)


def calculate_det_grad(X, net, llm=True):
    derivative_function = net_derivative
    a_omega = derivative_function(X.detach().clone(), net)
    # LOGGER.info(f"a_omega shape: {a_omega.shape}")
    n_preds = a_omega.shape[0]
    hidden_dim = a_omega.shape[1] if not llm else net.hidden_dim
    if llm:
        torch.manual_seed(0)
        a_omega = torch.randn(1, 1000, a_omega.shape[1]).cpu() @ a_omega
    else:
        a_omega = a_omega.reshape(n_preds, hidden_dim, -1)
    #
    # set random seed
    a_omega_dagger = torch.pinverse(a_omega).transpose(1, 2)
    probs = get_probs(net, X)  # [n_preds, seq_length, dict_size]

    g = a_omega_dagger @ probs.transpose(1, 2)
    if g.shape[-1] == 1:
        eigenvalues = torch.sqrt(torch.svd(g @ g.transpose(1, 2))[1])
    else:
        eigenvalues = torch.svd(g)[1]
    dets = torch.log(eigenvalues).sum(dim=1)
    return dets


def net_derivative(x, net, option=4, derivative=False):
    net.eval()
    # LOGGER.debug(f"model device: {x.device}")

    if option == 1:

        def _predict(z):
            return net(z).sum(dim=0)

        J = jacfwd(_predict)(x)
        J = J.transpose(2, 0).transpose(1, 3)
    elif option == 2:
        net.train()
        n_preds = x.shape[0]
        xp = x.clone().requires_grad_()
        preds = net(xp)
        latent_dim = x.shape[1]
        length_seq, dict_size = preds.shape[1:]
        # # # preds =preds.view(n_preds, length_seq * dict_size)
        J = torch.zeros((n_preds, latent_dim, length_seq, dict_size)).cpu()  # loop will fill in Jacobian
        for i in range(length_seq):
            for j in range(dict_size):
                grd = torch.autograd.grad(preds[:, i, j].sum(), xp, create_graph=True, retain_graph=True)[0]
                J[..., i, j] = grd.detach().cpu()

        J = J.reshape(n_preds, latent_dim, length_seq, dict_size)
    elif option == 3:
        eps = 1e-4
        latent_dim = x.shape[1]
        pred_x = net(x).detach().cpu()
        n_preds, length_seq, dict_size = pred_x.shape
        J = torch.zeros((n_preds, latent_dim, length_seq, dict_size)).cpu()  # loop will fill in Jacobian
        for i in range(latent_dim):
            dx = torch.zeros_like(x)
            dx[:, i] = eps
            dy = net(x + dx).detach().cpu() - pred_x
            J[:, i, :, :] = dy / eps
    elif option == 4:
        torch.cuda.empty_cache()
        eps = 1e-4
        batch_size, latent_dim = x.shape
        pred_x = net(x.clone()).detach().cpu()
        dxs_list = []
        pred_x = pred_x.repeat(latent_dim, 1, 1)
        for i in range(latent_dim):
            dx = torch.zeros_like(x)
            dx[:, i] = eps
            dxs_list.append(x.clone() + dx)
        del x
        dxs = torch.cat(dxs_list, dim=0)
        _batch_size = 128
        n_batches = int(np.ceil(dxs.shape[0] / _batch_size))
        dys_vec_batches = []

        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * _batch_size
            end_idx = (batch_idx + 1) * _batch_size

            # Get a batch of dxs
            dxs_batch = dxs[start_idx:end_idx]

            # Calculate dys for the batch
            dys_batch = net(dxs_batch).detach().cpu() - pred_x[start_idx:end_idx]

            dys_vec_batches.append(dys_batch)
        dys_vec = torch.cat(dys_vec_batches, dim=0)
        J = dys_vec / eps
        shift = np.arange(0, latent_dim * batch_size, batch_size)
        J = torch.stack([J[j + shift, ...] for j in range(batch_size)], dim=0)
        J = J.detach().cpu()

        J = J.squeeze()

        if len(J.shape) == 2:
            J = J.unsqueeze(0)

    else:
        raise ValueError(f"option {option} is not supported")
    if J.shape[-1] == 2 and not derivative:
        J = J[..., 0]
    return J
