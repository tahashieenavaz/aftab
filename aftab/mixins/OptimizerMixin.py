from types import SimpleNamespace


class OptimizerMixin:
    def __init__(self):
        super().__init__()

    def __build_categorical_optimizers(self):
        image_encoder_parameters = list(self._network.phi.parameters())
        quantile_value_parameters = list(self._network.quantile_value.parameters())
        fraction_proposal_parameters = list(
            self._network.fraction_proposal.parameters()
        )
        ordinary_parameters = image_encoder_parameters + quantile_value_parameters

        quantile_value_optimizer = self.optimizer_instance(
            ordinary_parameters,
            lr=self.lr,
            eps=self.optimizer_epsilon,
        )
        fraction_proposal_optimizer = self.optimizer_instance(
            fraction_proposal_parameters,
            lr=self.fraction_proposal_lr,
            eps=self.optimizer_epsilon,
        )

        return SimpleNamespace(
            **{
                "fraction_proposal": fraction_proposal_optimizer,
                "quantile_value": quantile_value_optimizer,
            }
        )

    def __build_non_categorical_optimizer(self):
        return self.optimizer_instance(
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )

    def make_optimizer(self):
        if self.network in ["q", "duelling"]:
            return self.__build_non_categorical_optimizer()
        else:
            return self.__build_categorical_optimizers()
