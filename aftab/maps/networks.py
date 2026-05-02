from ..networks import PQNNetwork
from ..networks import DuellingNetwork
from ..networks import BootstrappedNetwork
from ..networks import BootstrappedDuellingNetwork
from ..networks import DistributionalNetwork
from ..networks import DistributionalDuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "bootstrapped": BootstrappedNetwork,
    "bootstrapped-duelling": BootstrappedDuellingNetwork,
    "distributional": DistributionalNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
}
network_map = networks_map
