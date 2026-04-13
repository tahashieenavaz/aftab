import torch


class DummyPassMixin:
    def __init__(self):
        super().__init__()

    def get_dummy_sample(self):
        return torch.randn(1, 4, 84, 84).to(self.device)

    @torch.no_grad()
    def perform_dummy_pass(self):
        """
        This function helps the model to initialize all the lazy modules before training or compilation begins.
        """
        dummy_input = self.get_dummy_sample()
        self._network(dummy_input)
