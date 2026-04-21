import torch


def random_shifts(
    observation: torch.Tensor,
    height_shifts: torch.Tensor,
    width_shifts: torch.Tensor,
    *,
    padding: int,
):
    B, C, H, W = observation.shape
    device = observation.device
    padded = torch.nn.functional.pad(
        observation, (padding, padding, padding, padding), mode="replicate"
    )
    batch_idx = torch.arange(B, device=device)[:, None, None, None]
    c_idx = torch.arange(C, device=device)[None, :, None, None]
    h_idx = (
        torch.arange(H, device=device)[None, None, :, None]
        + height_shifts[:, None, None, None]
    )
    w_idx = (
        torch.arange(W, device=device)[None, None, None, :]
        + width_shifts[:, None, None, None]
    )
    return padded[batch_idx, c_idx, h_idx, w_idx]
