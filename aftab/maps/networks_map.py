from aftab.networks import PQNNetwork
from aftab.networks import DuellingNetwork
from aftab.networks import BootstrappedNetwork
from aftab.networks import BootstrappedDuellingNetwork
from aftab.networks import DistributionalNetwork
from aftab.networks import DistributionalDuellingNetwork
from aftab.networks import DistributionalBootstrappedDuellingNetwork
from aftab.networks import DistributionalBootstrappedDuellingMixedNetwork
from aftab.networks import DistributionalBootstrappedDuellingMixedDepthNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "bootstrapped": BootstrappedNetwork,
    "bootstrapped-duelling": BootstrappedDuellingNetwork,
    "distributional": DistributionalNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
    "distributional-bootstrapped-duelling": DistributionalBootstrappedDuellingNetwork,
    "bootstrapped-distributional-duelling": DistributionalBootstrappedDuellingNetwork,
    "distributional-bootstrapped-mixed-duelling": DistributionalBootstrappedDuellingMixedNetwork,
    "distributional-bootstrapped-mixed-depth-duelling": DistributionalBootstrappedDuellingMixedDepthNetwork,
    "d": DuellingNetwork,
    "bdd": DistributionalBootstrappedDuellingNetwork,
    "bd": BootstrappedDuellingNetwork,
    "dd": DistributionalDuellingNetwork,
    "b": BootstrappedNetwork,
}
