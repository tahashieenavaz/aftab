from ..agents import PQNAgent
from ..agents import DuellingAgent
from ..agents import FQFAgent

agents_map = {
    "regression": PQNAgent,
    "duelling": DuellingAgent,
    "fqf": FQFAgent,
}
agent_map = agents_map
