from ..networks import PQNNetwork
from ..networks import DuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
}
network_map = networks_map
