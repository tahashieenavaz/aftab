from .FQFNetwork import FQFNetwork
from ..modules import DuellingQuantileStream


class DuellingFQFNetwork(FQFNetwork):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.quantile_value = DuellingQuantileStream(
            action_dimension=self.action_dimension,
            feature_dimension=self.feature_dimension,
            embedding_dimension=kwargs["quantile_embedding_dimension"],
        )
