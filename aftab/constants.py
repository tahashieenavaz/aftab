import torch
from typing import Type
from typing import Literal


ModuleType = Type[torch.nn.Module]

EncoderStringType = Literal[
    "alpha",
    "beta",
    "gamma",
    "epsilon",
    "eta",
    "zeta",
    "delta",
    "theta",
    "hadamaxgammav1",
    "gammahadamaxv1",
    "hadamaxgammav2",
    "gammahadamaxv2",
    "nature",
]

OptimizerStringType = Literal["adam", "adamw" "radam", "nadam"]


seeds = [475284, 219842, 525975, 909314]

environments = [
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "Asteroids-v5",
    "Atlantis-v5",
    "BankHeist-v5",
    "BattleZone-v5",
    "BeamRider-v5",
    "Berzerk-v5",
    "Bowling-v5",
    "Boxing-v5",
    "Breakout-v5",
    "Centipede-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "Defender-v5",
    "DemonAttack-v5",
    "DoubleDunk-v5",
    "Enduro-v5",
    "FishingDerby-v5",
    "Freeway-v5",
    "Frostbite-v5",
    "Gopher-v5",
    "Gravitar-v5",
    "Hero-v5",
    "IceHockey-v5",
    "Jamesbond-v5",
    "Kangaroo-v5",
    "Krull-v5",
    "KungFuMaster-v5",
    "MontezumaRevenge-v5",
    "MsPacman-v5",
    "NameThisGame-v5",
    "Phoenix-v5",
    "Pitfall-v5",
    "Pong-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "Riverraid-v5",
    "RoadRunner-v5",
    "Robotank-v5",
    "Seaquest-v5",
    "Skiing-v5",
    "Solaris-v5",
    "SpaceInvaders-v5",
    "StarGunner-v5",
    "Surround-v5",
    "Tennis-v5",
    "TimePilot-v5",
    "Tutankham-v5",
    "UpNDown-v5",
    "Venture-v5",
    "VideoPinball-v5",
    "WizardOfWor-v5",
    "YarsRevenge-v5",
    "Zaxxon-v5",
]
