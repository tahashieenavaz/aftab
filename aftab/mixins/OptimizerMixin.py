class OptimizerMixin:
    def __init__(self):
        super().__init__()

    def make_optimizer(self):
        return self.optimizer(
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )
