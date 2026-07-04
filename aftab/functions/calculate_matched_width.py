import math


def calculate_matched_width(
    input_dim: int, target_hidden_dim: int, output_dim: int, depth: int, norm: bool
) -> int:
    if depth <= 2:
        return target_hidden_dim

    target_params = (input_dim * target_hidden_dim + target_hidden_dim) + (
        target_hidden_dim * output_dim + output_dim
    )
    if norm:
        target_params += 2 * target_hidden_dim
    a = depth - 2
    b = input_dim + output_dim + (depth - 2)
    if norm:
        b += 2 * (depth - 1)
    c = output_dim - target_params
    W = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return int(round(W))
