from aftab.networks import PQNNetwork
from aftab.networks import DuellingNetwork
from aftab.networks import BootstrappedNetwork
from aftab.networks import BootstrappedDuellingNetwork
from aftab.networks import DistributionalNetwork
from aftab.networks import DistributionalDuellingNetwork
from aftab.networks import DistributionalBootstrappedDuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "bootstrapped": BootstrappedNetwork,
    "bootstrapped-duelling": BootstrappedDuellingNetwork,
    "distributional": DistributionalNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
    "distributional-bootstrapped-duelling": DistributionalBootstrappedDuellingNetwork,
}
