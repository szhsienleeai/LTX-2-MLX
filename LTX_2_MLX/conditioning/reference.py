"""Reference-video conditioning for IC-LoRA inference."""

import mlx.core as mx

from LTX_2_MLX.types import LatentState, VideoLatentShape
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.conditioning.tools import VideoLatentTools


class VideoConditionByReferenceLatent:
    """
    Appends reference-video tokens as clean in-context tokens for IC-LoRA.

    Mirrors upstream ``ltx_core``'s ``VideoConditionByReferenceLatent``. IC-LoRAs
    can be trained with a reference smaller than the target (the pixel spatial
    upscaler uses a 1/2 or 1/4 reference); ``downscale_factor`` multiplies the
    reference's height/width positions so its tokens land on the target's
    coordinate grid, reproducing the positional relationship seen in training.
    With ``downscale_factor=1`` this matches ``VideoConditionByKeyframeIndex``
    at ``frame_idx=0``.
    """

    def __init__(self, latent: mx.array, downscale_factor: int = 1, strength: float = 1.0):
        """
        Args:
            latent: Encoded reference latents (B, C, F, H, W).
            downscale_factor: Target/reference spatial ratio (read from LoRA metadata).
            strength: 1.0 keeps the reference clean; 0.0 lets it be denoised.
        """
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)

        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_shape(self.latent.shape)
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        ).astype(mx.float32)

        temporal = positions[:, 0:1, ...] / latent_tools.fps
        spatial = positions[:, 1:, ...] * self.downscale_factor
        positions = mx.concatenate([temporal, spatial], axis=1)

        denoise_mask = mx.full(
            shape=(tokens.shape[0], tokens.shape[1], 1),
            vals=1.0 - self.strength,
            dtype=self.latent.dtype,
        )

        # Upstream puts zeros here because its noiser lerps from clean_latent where
        # mask=0; this fork's GaussianNoiser blends from `latent`, so zeros would
        # feed the transformer an empty reference.
        return LatentState(
            latent=mx.concatenate([latent_state.latent, tokens], axis=1),
            denoise_mask=mx.concatenate([latent_state.denoise_mask, denoise_mask], axis=1),
            positions=mx.concatenate([latent_state.positions, positions], axis=2),
            clean_latent=mx.concatenate([latent_state.clean_latent, tokens], axis=1),
        )
