import torch
import numpy as np


def loss_function_dc1(embedding, target):
    """
    Implement loss function (invalid)
    :param embedding: N * TF * Embedding Dim
    :param target: N * TF * 1 (vocal)
    :return: Loss value for one batch N * scalar
    """
    def create_diag(target_m):
        d_m = target_m.dot(target_m.T)
        d_m = np.sum(d_m, axis=1)
        return np.sqrt(np.diag(d_m))

    def f2_norm(matrix):
        r = np.linalg.norm(matrix)
        return r**2

    n, tf, dim = embedding.size()

    loss = 0

    for i in range(n):
        v = embedding[i].numpy()
        y = target[i].numpy()
        d = create_diag(y)
        loss += f2_norm((v.T.dot(d).dot(v))) - 2*f2_norm((v.T.dot(d).dot(y))) + f2_norm((y.T.dot(d).dot(y)))

    return loss


def loss_function_dc(embedding, target):
    """
    Implement loss function
    :param embedding: N * TF * Embedding Dim
    :param target: N * TF * 1 (vocal)
    :return: Loss value for one batch N * scalar
    """

    def l2_loss(x):
        norm = torch.norm(x, 2)
        return norm ** 2

    loss = l2_loss(torch.bmm(torch.transpose(embedding, 1, 2), embedding)) + \
        l2_loss(torch.bmm(torch.transpose(target, 1, 2), target)) - \
        l2_loss(torch.bmm(torch.transpose(embedding, 1, 2), target)) * 2

    return loss / torch.sum(target)


