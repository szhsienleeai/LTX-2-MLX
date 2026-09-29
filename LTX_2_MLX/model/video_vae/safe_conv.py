"""Batch-chunked mx.conv2d that stays under MLX's large-input miscompute."""

import mlx.core as mx

# mlx 0.30.1: mx.conv2d returns wrong values (max abs err ~9 on unit-scale
# data) once the input passes ~1.5e8 elements — batch 31 of (146, 258, 128)
# is exact, batch 33 is not. Conv2d is independent per batch entry, so
# splitting the batch is exact; 2**26 leaves >2x margin under the observed edge.
MAX_CONV2D_INPUT_ELEMS = 2**26


def conv2d_batched(x: mx.array, w: mx.array, padding: int = 0) -> mx.array:
    """mx.conv2d over (N, H, W, C) input, split along N to stay under the limit."""
    per_item = x.size // x.shape[0]
    step = max(1, MAX_CONV2D_INPUT_ELEMS // per_item)
    if x.shape[0] <= step:
        return mx.conv2d(x, w, padding=padding)
    return mx.concatenate(
        [mx.conv2d(x[i:i + step], w, padding=padding) for i in range(0, x.shape[0], step)],
        axis=0,
    )
