import torch
from .GammaHadamaxEncoder import GammaHadamaxEncoder


class GammaHadamaxReLUEncoder(GammaHadamaxEncoder):
    def __init__(self):
        super().__init__(activation=torch.nn.ReLU)
