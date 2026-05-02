from .AftabBaseMixin import AftabBaseMixin
from ..maps import optimizer_map


class AftabOptimizerMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def _initialize_optimizer(self):
        if self.optimizer not in optimizer_map:
            raise ValueError(f"Optimizer `{self.optimizer}` was not founded.")

        self._optimizer = optimizer_map[self.optimizer](
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )
