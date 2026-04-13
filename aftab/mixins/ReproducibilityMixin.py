from baloot import seed_everything


class ReproducibilityMixin:
    def __init__(self):
        super().__init__()

    def set_seed(self, seed: int):
        seed_everything(seed)
