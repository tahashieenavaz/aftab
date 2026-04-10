import torch
from baloot import acceleration_device


def mse_loss(A, B):
    _device = acceleration_device()
    A = A.to(_device)
    B = B.to(_device)
    return 0.5 * torch.nn.functional.mse_loss(A, B)
