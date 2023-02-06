import torch

def kl_divergence(x1, x2):
    mean_x1 = torch.mean(x1, dim=0)
    mean_x2 = torch.mean(x2, dim=0)
    cov_x1 = torch.mm((x1 - mean_x1).t(), x1 - mean_x1) / x1.size(0)
    cov_x2 = torch.mm((x2 - mean_x2).t(), x2 - mean_x2) / x2.size(0)
    return 0.5 * (torch.trace(cov_x2 @ torch.inverse(cov_x1)) + (mean_x2 - mean_x1).dot(torch.inverse(cov_x1) @ (mean_x2 - mean_x1)) - cov_x1.size(0) + torch.log(torch.det(cov_x1) / torch.det(cov_x2)))

