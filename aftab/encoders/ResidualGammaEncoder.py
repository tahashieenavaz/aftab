import torch
from ..common import LayerNorm2d, ModuleType

#####
# In this file I have used Persian letters for layers instead of naming them with numbers.
# alef: الف, be: ب, pe: پ, te: ت, se: ث
#####


class ResidualGammaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.alef = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(32),
            activation(),
        )
        self.be = torch.nn.Sequential(
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(48),
            activation(),
        )
        self.pe = torch.nn.Sequential(
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            LayerNorm2d(64),
            activation(),
        )
        self.te = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            LayerNorm2d(64),
            activation(),
        )
        self.se = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            LayerNorm2d(64),
            activation(),
        )

        self.alef_projection = torch.nn.Conv2d(4, 32, kernel_size=1, stride=2)
        self.be_projection = torch.nn.Conv2d(32, 48, kernel_size=1, stride=2)
        self.pe_projection = torch.nn.Conv2d(48, 64, kernel_size=1, stride=1)
        self.te_projection = torch.nn.Conv2d(64, 64, kernel_size=1, stride=2)
        self.se_projection = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1)

    def _crop(self, skip, target):
        _, _, th, tw = target.shape
        _, _, sh, sw = skip.shape
        top = (sh - th) // 2
        left = (sw - tw) // 2
        return skip[:, :, top : top + th, left : left + tw]

    def forward(self, x):
        x = self.alef(x) + self.alef_projection(x)

        r = x
        x = self.be(x) + self.be_projection(r)

        r = x
        pe_output = self.pe(x)
        x = pe_output + self.crop(self.pe_projection(r), pe_output)

        r = x
        te_output = self.te(x)
        x = te_output + self.crop(self.te_output(r), te_output)

        r = x
        se_output = self.se(x)
        x = se_output + self.crop(self.se_projection(r), se_output)

        return torch.flatten(x, 1)
