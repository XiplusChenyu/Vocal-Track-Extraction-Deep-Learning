import torch

def loss_function_dc(embedding, target):
    """
    Implement loss function
    :param embedding: N * TF * Embedding Dim
    :param target: N * TF * 1 (vocal)
    :return: Loss value for one batch N * scalar
    """

    def create_diag(target_m):
        """
        create dialog
        :param target_m: N * TF * 1 (vocal)
        :return: N * TF * TF
        """
        d_m = torch.bmm(target_m, torch.transpose(target_m, 1, 2))
        d_m = torch.sum(d_m, dim=2)  # notice there is batch
        d_m = torch.diag_embed(d_m)
        return torch.sqrt(d_m)

    def f2_norm(x):
        norm = torch.norm(x, 2)
        return norm ** 2

    diags = create_diag(target)
    n, tf, _ = embedding.shape
    part1 = f2_norm(torch.bmm(torch.bmm(torch.transpose(embedding, 1, 2), diags), embedding))
    part2 = f2_norm(torch.bmm(torch.bmm(torch.transpose(embedding, 1, 2), diags), target))
    part3 = f2_norm(torch.bmm(torch.bmm(torch.transpose(target, 1, 2), diags), target))

    return part1 - 2 * part2 + part3 / (n*tf)

