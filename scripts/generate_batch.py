#!/usr/bin/env python3
"""Multi-seed batch driver for generate.py — load weights ONCE, generate N seeds.

Motivation (vj-loop-pipeline seed sweeps): a single generate.py run spends most of
its wall time loading ~48GB of weights (Gemma text encoder + transformer); the
denoise itself is ~23s at 768x512x65f. Sweeping 3 seeds as 3 processes pays the
load 3x. This driver imports generate.py, memoizes the expensive loaders and the
prompt encoders at module level, then calls generate_video() once per seed:

  Gemma encode   1x per prompt   (memoized encode_with_gemma / encode_av_gemma_batch)
  transformer    1x per weights  (memoized load_transformer / load_av_transformer)
  VAE decode     per seed        (small; untouched)

Memory note: generate_video deliberately frees the transformer before VAE decode;
the memo cache keeps it alive instead. Peak RSS during decode therefore matches the
denoise-phase peak (transformer + activations), which this machine already sustains
— verified safe on M5 Max 64GB with the fp8 checkpoints. Do NOT combine with --lora
or cross-attn scaling: the cache would leak fused weights into later seeds.

Usage (from the repo root, same conventions as generate.py):
  python3 scripts/generate_batch.py "<prompt>" --seeds 7,21,88 \
    --pipeline distilled --weights weights/ltx-2/ltx-2-19b-distilled-fp8.safetensors \
    --fp8 --low-memory --height 512 --width 768 --frames 65 --steps 8 --cfg 1.0 \
    --fps 24 --output-template outputs/sweep_seed{seed}.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as G  # noqa: E402  (heavy import: mlx + model defs)


def _memoize(fn):
    cache = {}
    def wrapped(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        else:
            print(f"  [batch] cache hit: {fn.__name__} (skipping reload)")
        return cache[key]
    wrapped.__name__ = fn.__name__
    return wrapped


def main():
    ap = argparse.ArgumentParser(description="Multi-seed batch wrapper over generate.py")
    ap.add_argument("prompt")
    ap.add_argument("--seeds", required=True, help="Comma-separated seed list, e.g. 7,21,88")
    ap.add_argument("--pipeline", default="distilled")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--low-memory", action="store_true")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gemma-path", default="weights/gemma-3-12b")
    ap.add_argument("--output-template", required=True,
                    help="Per-seed output path with a {seed} placeholder")
    args = ap.parse_args()

    if "{seed}" not in args.output_template:
        raise SystemExit("--output-template must contain {seed}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds is empty")

    weights = args.weights
    if args.fp8 and ".safetensors" in weights and "-fp8" not in weights:
        weights = weights.replace(".safetensors", "-fp8.safetensors")

    # Memoize the expensive stages. Module-level patch: generate_video resolves
    # these names via module globals, so every internal call goes through the cache.
    G.load_transformer = _memoize(G.load_transformer)
    G.load_av_transformer = _memoize(G.load_av_transformer)
    G.encode_with_gemma = _memoize(G.encode_with_gemma)
    G.encode_av_gemma_batch = _memoize(G.encode_av_gemma_batch)

    print(f"[batch] {len(seeds)} seeds {seeds} | {args.width}x{args.height}x{args.frames}f "
          f"| pipeline={args.pipeline} | weights={weights}")
    for i, seed in enumerate(seeds, 1):
        out = args.output_template.format(seed=seed)
        print(f"\n[batch] ===== seed {seed} ({i}/{len(seeds)}) -> {out} =====")
        G.generate_video(
            prompt=args.prompt,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, cfg_scale=args.cfg, seed=seed,
            weights_path=weights, output_path=out,
            use_fp8=args.fp8, low_memory=args.low_memory,
            gemma_path=args.gemma_path,
            pipeline_type=args.pipeline, output_fps=args.fps,
        )
    print(f"\n[batch] done: {len(seeds)} seeds")


if __name__ == "__main__":
    main()
