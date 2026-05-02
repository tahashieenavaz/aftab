import torch
from ..maps import networks_map

_DISTRIBUTIONAL_NETWORKS = {"distributional", "distributional-duelling"}
_BOOTSTRAPPED_NETWORKS = {"bootstrapped", "bootstrapped-duelling"}


class NetworkMixin:
    def __init__(self):
        super().__init__()

    def __channels_last_enabled(self) -> bool:
        return bool(getattr(self, "channels_last"))

    def __dummy_input(self) -> torch.Tensor:
        if not hasattr(self, "frame_stack"):
            raise AttributeError("Expected `frame_stack` to be defined.")
        if not hasattr(self, "device"):
            raise AttributeError("Expected `device` to be defined.")

        sample = torch.randn(
            1,
            self.frame_stack,
            84,
            84,
            device=self.device,
        )
        if self.__channels_last_enabled():
            return sample.contiguous(memory_format=torch.channels_last)
        return sample

    def __distributional_sigma(self) -> float:
        sigma = getattr(self, "distributional_sigma", None)
        if sigma is None:
            bins = int(getattr(self, "distributional_bins"))
            min_value = float(getattr(self, "distributional_min_value"))
            max_value = float(getattr(self, "distributional_max_value"))
            if bins <= 0:
                raise ValueError("Expected `distributional_bins` to be positive.")
            if max_value <= min_value:
                raise ValueError(
                    "Expected `distributional_max_value` to be greater than "
                    "`distributional_min_value`."
                )
            bin_width = (max_value - min_value) / bins
            sigma = bin_width * float(getattr(self, "distributional_sigma_ratio"))
            sigma_name = "`distributional_sigma_ratio`"
        else:
            sigma = float(sigma)
            sigma_name = "`distributional_sigma`"

        if sigma <= 0.0:
            raise ValueError(f"Expected {sigma_name} to be positive.")
        return sigma

    def __network_kwargs(self) -> dict:
        kwargs = {}
        if self.network in _DISTRIBUTIONAL_NETWORKS:
            kwargs.update(
                distributional_bins=int(getattr(self, "distributional_bins")),
                distributional_min_value=float(
                    getattr(self, "distributional_min_value")
                ),
                distributional_max_value=float(
                    getattr(self, "distributional_max_value")
                ),
                distributional_sigma=self.__distributional_sigma(),
            )

        if self.network not in _BOOTSTRAPPED_NETWORKS:
            return kwargs

        bootstrap_heads = int(getattr(self, "bootstrap_heads"))
        if bootstrap_heads <= 0:
            raise ValueError("Expected `bootstrap_heads` to be positive.")
        kwargs["bootstrap_heads"] = bootstrap_heads

        return kwargs

    def prepare_network(self, action_dimension: int):
        try:
            network_instance = networks_map[self.network]
        except KeyError as exc:
            raise ValueError(
                f"Invalid value for `network`: {self.network!r}. "
                f"Expected one of {tuple(networks_map)}."
            ) from exc

        self.flush_verbose(f"Network: {network_instance.__name__}")
        self._network = network_instance(
            action_dimension=action_dimension,
            embedding_dimension=self.embedding_dimension,
            encoder=self.encoder,
            channels_last=self.__channels_last_enabled(),
            **self.__network_kwargs(),
        )
        if self.__channels_last_enabled():
            self._network.to(device=self.device, memory_format=torch.channels_last)
        else:
            self._network.to(self.device)

        with torch.no_grad():
            self._network(self.__dummy_input())

        if bool(getattr(self, "torch_compile")):
            self._network = torch.compile(self._network)
