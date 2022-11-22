import numpy as np
import torch
from scipy.optimize import nnls
from torch import optim
from tqdm import tqdm

from model.UFSTAE import AutoEncoder
from utils.clusteringPerformance import StatisticClustering


def train(x, y, k, criterion, epochs, alpha, beta, lr, lr_milestones, device):
    x = x.to(device)
    n, d = x.shape[0], x.shape[1]
    model = AutoEncoder(d, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=lr_milestones, gamma=0.1)
    best_ACC = -1.
    ACC = -1.
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        optimizer.zero_grad()
        middle, output = model(x)
        loss = torch.sum(criterion(output, x), dim=1)

        # Add the value of the regularization term to the loss
        reg_loss = torch.ones_like(loss) * regularization(model.encoder, alpha)
        ortho_loss = torch.ones_like(loss) * orthogonality_loss(model.encoder, beta, device)

        loss = loss + reg_loss + ortho_loss
        total_loss = torch.mean(loss)
        loss.backward(gradient=torch.ones_like(loss))
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            middle, _ = model(x)

            # Find a non-negative indicator matrix through the middle and the initial input x
            w = np.asmatrix(np.zeros((d, k)))
            for i in range(k):
                b = np.array((middle.cpu().detach().numpy())[:, i]).flatten()
                temp_input = x.cpu().numpy()
                t, residual = nnls(temp_input, b)
                w[:, i] = np.asmatrix(t).transpose()
            w = np.asmatrix(w).transpose()
            index = np.argsort(-(np.sum(np.array(np.square(w)), axis=0)), axis=0)  # Sort in descending order
            w_ = index[0:k]
            # print(w_)

            # Evaluation
            ACC, NMI, ARI = StatisticClustering(x[:, w_].cpu().detach().numpy(), y, 'KMeans')
            if best_ACC < ACC[0]:
                best_ACC = ACC[0]
            model.train()
        pbar.set_description("Epoch{}| #Selected Features {}, Loss: {:.4}, ACC{}, Best ACC{}".format(
            epoch,
            k,
            total_loss.item(),
            ACC,
            best_ACC)
        )

    return ACC, NMI, ARI


def regularization(model, weight_decay):
    reg_loss = 0
    if hasattr(model[0], 'weight'):
        w = model[0].weight.data.t()
        l21_reg = torch.norm(w, p=2, dim=1)
        l21_reg = torch.norm(l21_reg, p=1)
        reg_loss = weight_decay * l21_reg
    return reg_loss


def orthogonality_loss(model, a, device):
    o_loss = 0
    if hasattr(model[0], 'weight'):
        w = model[0].weight.data
        k, d = w.shape
        I = torch.eye(k).to(device)
        o_loss += a * torch.norm(torch.sub(w.matmul(w.t()), I), p='fro')  # Orthogonality Constraint
    else:
        pass
    return o_loss
