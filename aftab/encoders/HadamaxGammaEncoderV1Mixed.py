from aftab.modules import MixedActivation
from .HadamaxGammaEncoderV1 import HadamaxGammaEncoderV1


class HadamaxGammaEncoderV1Mixed(HadamaxGammaEncoderV1):
    def __init__(self):
        super().__init__(activation=MixedActivation)
