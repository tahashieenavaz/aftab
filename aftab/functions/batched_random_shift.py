import torch


def batched_random_shift(batches_observations, padding: int = 4):
    n, c, h, w = batches_observations.shape
    padded = torch.nn.functional.pad(
        batches_observations, (padding, padding, padding, padding), mode="replicate"
    )
    cropped = torch.empty_like(batches_observations)
    w_starts = torch.randint(0, 2 * padding + 1, (n,))
    h_starts = torch.randint(0, 2 * padding + 1, (n,))
    for i in range(n):
        cropped[i] = padded[
            i, :, h_starts[i] : h_starts[i] + h, w_starts[i] : w_starts[i] + w
        ]
    return cropped
