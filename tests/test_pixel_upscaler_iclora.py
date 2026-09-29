"""Pixel spatial upscaler IC-LoRA support: downscaled reference + in-place fusion."""

import json
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from LTX_2_MLX.components.patchifiers import VideoLatentPatchifier
from LTX_2_MLX.conditioning.keyframe import VideoConditionByKeyframeIndex
from LTX_2_MLX.conditioning.reference import VideoConditionByReferenceLatent
from LTX_2_MLX.conditioning.tools import VideoLatentTools
from LTX_2_MLX.loader import LoRAConfig, fuse_lora_streaming, read_lora_metadata
from LTX_2_MLX.types import VideoLatentShape


def _tools(frames=3, h=8, w=12):
    shape = VideoLatentShape(batch=1, channels=128, frames=frames, height=h, width=w)
    return VideoLatentTools(patchifier=VideoLatentPatchifier(patch_size=1), target_shape=shape, fps=24.0)


def test_reference_factor_1_matches_keyframe_frame0():
    tools = _tools()
    state = tools.create_initial_state()
    ref = mx.random.normal((1, 128, 3, 8, 12))
    a = VideoConditionByReferenceLatent(ref, downscale_factor=1, strength=1.0).apply_to(state, tools)
    b = VideoConditionByKeyframeIndex(ref, frame_idx=0, strength=1.0).apply_to(state, tools)
    for x, y in [(a.positions, b.positions), (a.clean_latent, b.clean_latent),
                 (a.latent, b.latent), (a.denoise_mask, b.denoise_mask)]:
        assert np.allclose(np.array(x), np.array(y))


def test_downscaled_reference_lands_on_target_grid():
    tools = _tools(frames=3, h=8, w=12)
    state = tools.create_initial_state()
    n_target = state.latent.shape[1]
    ref = mx.random.normal((1, 128, 3, 4, 6))  # half-size reference
    out = VideoConditionByReferenceLatent(ref, downscale_factor=2).apply_to(state, tools)

    ref_pos = np.array(out.positions[:, :, n_target:, :])
    tgt_pos = np.array(state.positions)
    assert out.latent.shape[1] == n_target + 3 * 4 * 6
    # spatial extent of the scaled reference equals the target's
    for axis in (1, 2):
        assert ref_pos[:, axis].min() == tgt_pos[:, axis].min()
        assert ref_pos[:, axis].max() == tgt_pos[:, axis].max()
    # time axis untouched
    assert np.allclose(ref_pos[:, 0].max(), tgt_pos[:, 0].max())
    # reference tokens are fed to the transformer (this fork's noiser blends from `latent`)
    assert np.allclose(np.array(out.latent[:, n_target:]), np.array(out.clean_latent[:, n_target:]))
    assert np.all(np.array(out.denoise_mask[:, n_target:]) == 0.0)


def _write_safetensors(path, tensors, metadata):
    header, blobs, off = {"__metadata__": metadata}, [], 0
    for name, arr in tensors.items():
        b = arr.astype(np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)
        self.other = nn.Linear(4, 3, bias=False)


def test_read_metadata_and_streaming_fuse(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 4)).astype(np.float32)
    B = rng.standard_normal((3, 2)).astype(np.float32)
    lora = tmp_path / "lora.safetensors"
    _write_safetensors(lora, {"proj.lora_A.weight": A, "proj.lora_B.weight": B},
                       {"reference_downscale_factor": "2"})
    assert read_lora_metadata(str(lora))["reference_downscale_factor"] == "2"

    model = _Tiny()
    mx.eval(model.parameters())
    w_proj = np.array(model.proj.weight)
    w_other = np.array(model.other.weight)

    n = fuse_lora_streaming(model, [LoRAConfig(path=str(lora), strength=0.5)], verbose=False)
    assert n == 1
    # Metal fp32 matmul carries ~2e-3 relative error; the real weights are
    # bf16 (eps 7.8e-3), so that is below what the model can represent anyway.
    assert np.allclose(np.array(model.proj.weight) - w_proj, 0.5 * (B @ A), rtol=5e-3, atol=1e-3)
    assert np.array_equal(np.array(model.other.weight), w_other)


def test_streaming_fuse_refuses_unmatched_lora(tmp_path):
    lora = tmp_path / "bad.safetensors"
    _write_safetensors(lora, {"nope.lora_A.weight": np.ones((2, 4)), "nope.lora_B.weight": np.ones((3, 2))}, {})
    with pytest.raises(RuntimeError, match="matched no model weights"):
        fuse_lora_streaming(_Tiny(), [LoRAConfig(path=str(lora))], verbose=False)


def test_lora_keys_follow_converter_renames():
    from LTX_2_MLX.loader.lora_loader import normalize_lora_key as n
    assert n("diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight") == \
        "transformer_blocks.0.attn1.to_out.lora_A.weight"
    assert n("diffusion_model.transformer_blocks.3.ff.net.0.proj.lora_B.weight") == \
        "transformer_blocks.3.ff.project_in.proj.lora_B.weight"
    assert n("diffusion_model.transformer_blocks.3.ff.net.2.lora_A.weight") == \
        "transformer_blocks.3.ff.project_out.lora_A.weight"


def test_streaming_fuse_refuses_partial_lora(tmp_path):
    lora = tmp_path / "partial.safetensors"
    _write_safetensors(lora, {
        "proj.lora_A.weight": np.ones((2, 4)), "proj.lora_B.weight": np.ones((3, 2)),
        "missing.lora_A.weight": np.ones((2, 4)), "missing.lora_B.weight": np.ones((3, 2)),
    }, {})
    with pytest.raises(RuntimeError, match="matched no model weight"):
        fuse_lora_streaming(_Tiny(), [LoRAConfig(path=str(lora))], verbose=False)


def test_conv2d_batched_matches_single_call(monkeypatch):
    import LTX_2_MLX.model.video_vae.safe_conv as sc
    x = mx.random.normal((7, 12, 10, 8))
    w = mx.random.normal((5, 3, 3, 8))
    ref = mx.conv2d(x, w)
    monkeypatch.setattr(sc, "MAX_CONV2D_INPUT_ELEMS", 12 * 10 * 8 * 2)  # force 2-item chunks
    out = sc.conv2d_batched(x, w)
    assert out.shape == ref.shape
    assert np.allclose(np.array(out), np.array(ref), atol=1e-4)
