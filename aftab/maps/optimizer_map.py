import torch

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "radam": torch.optim.RAdam,
    "nadam": torch.optim.NAdam,
}
