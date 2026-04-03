from ..encoders import AlphaEncoder
from ..encoders import BetaEncoder
from ..encoders import GammaEncoder
from ..encoders import DeltaEncoder
from ..encoders import EpsilonEncoder
from ..encoders import EtaEncoder
from ..encoders import ZetaEncoder
from ..encoders import ThetaEncoder
from ..encoders import GammaHadamaxEncoder

AftabMapEncoder = {
    "alpha": AlphaEncoder,
    "beta": BetaEncoder,
    "gamma": GammaEncoder,
    "delta": DeltaEncoder,
    "epsilon": EpsilonEncoder,
    "eta": EtaEncoder,
    "zeta": ZetaEncoder,
    "theta": ThetaEncoder,
    "gammahadamax": GammaHadamaxEncoder,
    "hadamaxgamma": GammaHadamaxEncoder,
}
