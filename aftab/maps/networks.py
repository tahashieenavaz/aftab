from ..networks import PQNNetwork, DuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
}
network_map = networks_map
