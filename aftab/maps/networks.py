from ..networks import PQNNetwork
from ..networks import DuellingNetwork
from ..networks import DistributionalPQNNetwork
from ..networks import DistributionalDuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "distributional": DistributionalPQNNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
}
network_map = networks_map
