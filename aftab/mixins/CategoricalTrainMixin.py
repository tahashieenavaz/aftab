import torch
import numpy
import time
from ..functions import flush, lambda_returns


class CategoricalTrainMixin:
    def __init__(self):
        super().__init__()

    def categorical_train(self, environment: str, seed: int):
        pass
