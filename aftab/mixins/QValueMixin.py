import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def __get_q_value_and_quantiles(
        self, float_observations: torch.Tensor, gradient: bool
    ):
        with (
            torch.set_grad_enabled(gradient),
            torch.autocast(device_type=self.device.type, dtype=torch.float16),
        ):
            features = self._network.phi(float_observations)
            _, tau_hat, q_probs, _ = self._network.fraction_proposal(features)
            quantiles = self._network.quantile_value(features, tau_hat)
            q_values = (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)
            return {"q_values": q_values, "quantiles": quantiles}

    def get_q_values(
        self,
        float_train_observations: torch.Tensor = None,
        float_test_observations: torch.Tensor = None,
        float_observations: torch.Tensor = None,
        gradient: bool = False,
    ):
        with torch.set_grad_enabled(gradient):
            if float_observations is not None:
                return self._network(float_observations)

            test_q_values = self._network(float_test_observations)

            if getattr(self, "augmentation", "none") in ["none", "off"]:
                train_q_values = self._network(float_train_observations)
            else:
                augmentation_iterations = getattr(self, "augmentation_iterations")
                q_values_list = []
                for _ in range(augmentation_iterations):
                    augmented_train_observations = self.augmentation_pipeline(
                        float_train_observations
                    )
                    q_values_list.append(self._network(augmented_train_observations))
                train_q_values = torch.stack(q_values_list).mean(dim=0)

            return {"test": test_q_values, "train": train_q_values}

    def get_q_and_quantiles(
        self,
        float_train_observations: torch.Tensor = None,
        float_test_observations: torch.Tensor = None,
        float_observations: torch.Tensor = None,
        gradient: bool = False,
    ):
        if float_observations is not None:
            res = self.__get_q_value_and_quantiles(float_observations, gradient)
            return res["q_values"], res["quantiles"]

        test_res = self.__get_q_value_and_quantiles(float_test_observations, gradient)
        test_q_values = test_res["q_values"]
        test_quantiles = test_res["quantiles"]

        if getattr(self, "augmentation", "none") in ["none", "off"]:
            train_res = self.__get_q_value_and_quantiles(
                float_train_observations, gradient
            )
            train_q_values = train_res["q_values"]
            train_quantiles = train_res["quantiles"]
        else:
            augmentation_iterations = getattr(self, "augmentation_iterations")
            q_values_list = []
            quantiles_list = []
            for _ in range(augmentation_iterations):
                augmented_train_observations = self.augmentation_pipeline(
                    float_train_observations
                )
                res = self.__get_q_value_and_quantiles(
                    float_observations=augmented_train_observations, gradient=gradient
                )
                q_values_list.append(res["q_values"])
                quantiles_list.append(res["quantiles"])
            train_q_values = torch.stack(q_values_list).mean(dim=0)
            train_quantiles = torch.stack(quantiles_list).mean(dim=0)

        q_values = {"train": train_q_values, "test": test_q_values}
        quantiles = torch.cat([train_quantiles, test_quantiles], dim=0)
        return q_values, quantiles
