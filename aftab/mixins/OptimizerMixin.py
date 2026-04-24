import torch
from types import SimpleNamespace


class OptimizerMixin:
    def __init__(self):
        super().__init__()

    def __uses_adam_fraction_optimizer(self):
        optimizer_name = getattr(self.optimizer_instance, "__name__", "")
        return "adam" in optimizer_name.lower()

    def __build_fraction_proposal_optimizer(self, fraction_proposal_parameters):
        if self.__uses_adam_fraction_optimizer():
            return torch.optim.Adam(
                fraction_proposal_parameters,
                lr=self.fraction_proposal_lr,
                eps=0.0003125,
            )

        return torch.optim.RMSprop(
            fraction_proposal_parameters,
            lr=self.fraction_proposal_lr,
            alpha=0.95,
            momentum=0.0,
            eps=0.00001,
            centered=True,
        )

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
        fraction_proposal_optimizer = self.__build_fraction_proposal_optimizer(
            fraction_proposal_parameters
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
