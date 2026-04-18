from ..networks import PQNNetwork, DuellingNetwork, FQFNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "fqf": FQFNetwork,
}
network_map = networks_map
