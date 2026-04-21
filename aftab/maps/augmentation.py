import torch
from ..modules.augmentation import RandomShift
from ..modules.augmentation import ColorIntensity

augmentation_map = {
    "none": torch.nn.Identity(),
    "off": torch.nn.Identity(),
    "shift": RandomShift(),
    "randomshift": RandomShift(),
    "cut": RandomShift(),
    "intensity": ColorIntensity(),
    "color": ColorIntensity(),
    "all": torch.nn.Sequential(RandomShift(), ColorIntensity()),
}
