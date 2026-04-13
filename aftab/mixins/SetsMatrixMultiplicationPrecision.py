import torch


class SetsMatrixMultiplicationPrecision:
    def __init__(self):
        super().__init__()

    def set_precision(self):
        torch.set_float32_matmul_precision("high")
