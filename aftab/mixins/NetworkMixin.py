import torch
from ..maps import networks_map
from ..constants import ModuleType


class NetworkMixin:
    def __init__(self):
        super().__init__()

    def __channels_last_enabled(self) -> bool:
        return bool(getattr(self, "channels_last"))

    def __as_channels_last(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.__channels_last_enabled() or tensor.ndim != 4:
            return tensor
        return tensor.contiguous(memory_format=torch.channels_last)

    def __get_dummy_sample(self):
        if not hasattr(self, "frame_stack"):
            raise AttributeError("Expected `frame_stack` to be defined.")

        if not hasattr(self, "device"):
            raise AttributeError("Expected `device` to be defined.")

        batch_size = 1
        picture_size = 84

        sample = torch.randn(
            batch_size,
            self.frame_stack,
            picture_size,
            picture_size,
            device=self.device,
        )
        return self.__as_channels_last(sample)

    @torch.no_grad()
    def __perform_dummy_pass(self):
        if not hasattr(self, "_network"):
            raise AttributeError("Expected `_network` to be defined.")

        dummy_input = self.__get_dummy_sample()
        self._network(dummy_input)

    def __build_pqn_network(
        self,
        action_dimension: int,
        embedding_dimension: int,
        network_instance: ModuleType,
        **network_kwargs,
    ):
        self.flush_verbose(f"Network: {network_instance.__name__}")
        self._network = network_instance(
            action_dimension=action_dimension,
            embedding_dimension=embedding_dimension,
            encoder=self.encoder,
            channels_last=self.__channels_last_enabled(),
            **network_kwargs,
        )

    def __get_distributional_sigma(self) -> float:
        sigma = getattr(self, "distributional_sigma", None)
        if sigma is not None:
            sigma = float(sigma)
            if sigma <= 0.0:
                raise ValueError("Expected `distributional_sigma` to be positive.")
            return sigma

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
        if sigma <= 0.0:
            raise ValueError("Expected `distributional_sigma_ratio` to be positive.")
        return sigma

    def __get_distributional_network_args(self):
        return {
            "distributional_bins": int(getattr(self, "distributional_bins")),
            "distributional_min_value": float(
                getattr(self, "distributional_min_value")
            ),
            "distributional_max_value": float(
                getattr(self, "distributional_max_value")
            ),
            "distributional_sigma": self.__get_distributional_sigma(),
        }

    def __build_network(self, action_dimension: int):
        try:
            network_instance = networks_map[self.network]
        except KeyError as exc:
            raise ValueError(
                f"Invalid value for `network`: {self.network!r}. "
                f"Expected one of {tuple(networks_map)}."
            ) from exc

        args = {
            "network_instance": network_instance,
            "action_dimension": action_dimension,
            "embedding_dimension": self.embedding_dimension,
        }
        if self.network in {"distributional", "distributional-duelling"}:
            args.update(self.__get_distributional_network_args())
        self.__build_pqn_network(**args)

    def __compile_network(self):
        if not bool(getattr(self, "should_compile")):
            return

        if not hasattr(self, "_network"):
            raise AttributeError(
                "Expected `_network` to be defined before compilation."
            )

        self._network = torch.compile(self._network)

    def __move_network_on_device(self):
        if self.__channels_last_enabled():
            self._network.to(device=self.device, memory_format=torch.channels_last)
            return

        self._network.to(self.device)

    def prepare_network(self, action_dimension: int):
        self.__build_network(action_dimension=action_dimension)
        self.__move_network_on_device()
        self.__perform_dummy_pass()
        self.__compile_network()
