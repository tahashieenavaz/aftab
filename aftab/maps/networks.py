from ..networks import PQNNetwork, DuellingNetwork, FQFNetwork, DuellingFQFNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "fqf": FQFNetwork,
    "dfqf": DuellingFQFNetwork,
}
network_map = networks_map
