import torch
from ..maps import agents_map
from ..agents import PQNAgent
from ..constants import ModuleType


class NetworkMixin:
    def __init__(self):
        super().__init__()

    def __get_dummy_sample(self):
        if not hasattr(self, "stack_number"):
            raise AttributeError("Expected `stack_number` to be defined.")

        if not hasattr(self, "device"):
            raise AttributeError("Expected `device` to be defined.")

        batch_size = 1
        picture_size = 84

        return torch.randn(
            batch_size,
            self.stack_number,
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

    def __build_pqn_agent(self, action_dimension: int, agent_instance: ModuleType):
        self._network = agent_instance(
            action_dimension=action_dimension, encoder_instance=self.encoder
        )

    def __build_categorical_agent(
        self, action_dimension: int, agent_instance: ModuleType
    ):
        self._network = agent_instance(
            action_dimension=action_dimension,
            encoder_instance=self.encoder,
            number_quantiles=self.number_quantiles,
            embedding_dimension=self.embedding_dimension,
        )

    def __build_network(self, action_dimension: int):
        try:
            agent_instance = agents_map[self.agent]
            if agent_instance == PQNAgent:
                self.__build_pqn_agent(
                    action_dimension=action_dimension, agent_instance=agent_instance
                )
            else:
                self.__build_categorical_agent(
                    action_dimension=action_dimension, agent_instance=agent_instance
                )
        except:
            raise ValueError("Wrong strategy detected.")

    def __compile_network(self):
        if not getattr(self, "should_compile", False):
            return

        if not hasattr(self, "_network"):
            raise AttributeError(
                "Expected `_network` to be defined before compilation."
            )

        self._network = torch.compile(self._network)

    def prepare_network(self, action_dimension: int):
        self.__build_network(action_dimension=action_dimension)
        self.__perform_dummy_pass()
        self.__compile_network()
