from aftab.encoders import NatureDQNEncoder
from aftab.encoders import AlphaEncoder
from aftab.encoders import BetaEncoder
from aftab.encoders import GammaEncoder
from aftab.encoders import DeltaEncoder
from aftab.encoders import EpsilonEncoder
from aftab.encoders import EtaEncoder
from aftab.encoders import ZetaEncoder
from aftab.encoders import ThetaEncoder
from aftab.encoders import HadamaxNatureDQNEncoder
from aftab.encoders import HadamaxGammaEncoderV1
from aftab.encoders import HadamaxGammaEncoderV2
from aftab.encoders import HadamaxEpsilonEncoder
from aftab.encoders import HadamaxZetaEncoder
from aftab.encoders import HadamaxDeltaEncoder
from aftab.encoders import HadamaxGammaEncoderV1Mixed

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
    "hadamaxepsilon": HadamaxEpsilonEncoder,
    "epsilonhadamax": HadamaxEpsilonEncoder,
    "hadamaxzeta": HadamaxZetaEncoder,
    "zetahadamax": HadamaxZetaEncoder,
    "hadamaxdelta": HadamaxDeltaEncoder,
    "deltahadamax": HadamaxDeltaEncoder,
    "hadamaxgammav1mixed": HadamaxGammaEncoderV1Mixed,
    "gammahadamaxv1mixed": HadamaxGammaEncoderV1Mixed,
}
