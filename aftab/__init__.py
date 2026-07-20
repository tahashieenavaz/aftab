from .Aftab import Aftab

from .constants import seeds
from .constants import seeds as SEEDS
from .constants import seeds as aftab_seeds

from .constants import atari_environments
from .constants import environments
from .constants import procgen_environments
from .constants import atari_environments as ATARI_ENVS
from .constants import procgen_environments as PROCGEN_ENVS
from .constants import environments as ENVS
from .constants import environments as aftab_environments

from importlib.metadata import version

__version__ = version("aftab")
