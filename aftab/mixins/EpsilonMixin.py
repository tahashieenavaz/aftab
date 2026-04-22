import torch


class EpsilonMixin:
    def __init__(self):
        super().__init__()

    def get_epsilons(self, epsilon_value) -> torch.Tensor:
        training_epsilon_vector = torch.full(
            (self.train_environments,),
            epsilon_value,
            device=self.device,
            dtype=torch.float32,
        )
        test_epsilon_vector = torch.zeros(
            (self.test_environments,),
            device=self.device,
            dtype=torch.float32,
        )
        return torch.cat([training_epsilon_vector, test_epsilon_vector])
