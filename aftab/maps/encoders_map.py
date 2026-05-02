from ..encoders import NatureDQNEncoder
from ..encoders import AlphaEncoder
from ..encoders import BetaEncoder
from ..encoders import GammaEncoder
from ..encoders import DeltaEncoder
from ..encoders import EpsilonEncoder
from ..encoders import EtaEncoder
from ..encoders import ZetaEncoder
from ..encoders import ThetaEncoder
from ..encoders import HadamaxNatureDQNEncoder
from ..encoders import HadamaxGammaEncoderV1
from ..encoders import HadamaxGammaEncoderV2

encoders_map = {
    "nature": NatureDQNEncoder,
    "dqn": NatureDQNEncoder,
    "alpha": AlphaEncoder,
    "beta": BetaEncoder,
    "gamma": GammaEncoder,
    "delta": DeltaEncoder,
    "epsilon": EpsilonEncoder,
    "eta": EtaEncoder,
    "zeta": ZetaEncoder,
    "theta": ThetaEncoder,
    "hadamax": HadamaxNatureDQNEncoder,
    "dqnhadamax": HadamaxNatureDQNEncoder,
    "pqnhadamax": HadamaxNatureDQNEncoder,
    "hadamaxgammav1": HadamaxGammaEncoderV1,
    "gammahadamaxv1": HadamaxGammaEncoderV1,
    "hadamaxgammav2": HadamaxGammaEncoderV2,
    "gammahadamaxv2": HadamaxGammaEncoderV2,
}
