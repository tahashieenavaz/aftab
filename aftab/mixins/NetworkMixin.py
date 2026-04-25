import torch
from ..maps import networks_map
from ..constants import ModuleType


class NetworkMixin:
    def __init__(self):
        super().__init__()

    def __get_dummy_sample(self):
        if not hasattr(self, "frame_stack"):
            raise AttributeError("Expected `frame_stack` to be defined.")

        if not hasattr(self, "device"):
            raise AttributeError("Expected `device` to be defined.")

        batch_size = 1
        picture_size = 84

        return torch.randn(
            batch_size,
            self.frame_stack,
            picture_size,
            picture_size,
            device=self.device,
        )

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
    ):
        self._network = network_instance(
            action_dimension=action_dimension,
            embedding_dimension=embedding_dimension,
            encoder=self.encoder,
        )

    def __build_network(self, action_dimension: int):
        try:
            network_instance = networks_map[self.network]
            args = {
                "network_instance": network_instance,
                "action_dimension": action_dimension,
                "embedding_dimension": self.embedding_dimension,
            }
            self.__build_pqn_network(**args)
        except Exception as e:
            raise ValueError("Wrong network id detected.", e)

    def __compile_network(self):
        if not getattr(self, "should_compile"):
            return

        if not hasattr(self, "_network"):
            raise AttributeError(
                "Expected `_network` to be defined before compilation."
            )

        self._network = torch.compile(self._network)

    def __move_network_on_device(self):
        self._network.to(self.device)

    def prepare_network(self, action_dimension: int):
        self.__build_network(action_dimension=action_dimension)
        self.__move_network_on_device()
        self.__perform_dummy_pass()
        self.__compile_network()
