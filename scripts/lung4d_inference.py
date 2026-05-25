"""Run the DRR→CT 2D-4D model on every frame of every paired cycle.

For each case in ``outputs/lung4d_input/paired_dataset.npz`` we iterate
the 32 phase frames, feed ``(AP_t, LAT_t)`` through the model, and stack
the resulting CT volumes into a 4D tensor ``(T, Z, Y, X)`` which is the
generated 4DCT for that breath.

The model lives outside this repo at
``/Users/atuu/Documents/New project/model_20260428``; we import its
classes by adding that directory to ``sys.path``. The checkpoint is the
epoch-171 "large model" file in the same directory.

Defaults match ``visualization_simple_v2_cross_attn.py``'s argparse
defaults; override via CLI flags. ``--do-sr`` upscales 128³ → 256³.

Notes
-----
- ``ModelMP3GPU`` is multi-GPU by design. Setting ``--device`` makes all
  three stages live on the same device (cuda / mps / cpu).
- ``--amp`` is ignored unless the device is CUDA (autocast hardcoded to
  device_type="cuda" inside the model).
- Outputs go to ``outputs/lung4d_inference/case_<id>/`` and are
  patient-derived → gitignored.

Performance reality
-------------------
Wall-clock per frame, measured locally with the epoch-171 large ckpt:

  - CPU (Mac):        impractically slow (≥30 min / frame, did not finish)
  - MPS (M-series):   ~18 min / frame
  - CUDA (single GPU + AMP): expected ~1–10 s / frame (server-class)

14 cases × 32 frames × 18 min on MPS ≈ 150 hours — not realistic locally.
Recommended to run on the GPU server where the model was trained; this
script just orchestrates the inputs and outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


MODEL_DIR = Path("/Users/atuu/Documents/New project/model_20260428")
DEFAULT_CKPT = MODEL_DIR / "model_v2_mod_multiscale_L1_add_perpectual2_large_model_epoch171_f.pt"


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _install_dependency_stubs() -> None:
    """The training script imports two helper modules from the AMED data-prep
    pipeline (``create_DRR_AMED_original_pt_ver_mod``, ``DRR_norm_percentile``)
    purely for the DRR-consistency loss code path. None of that is exercised
    during a pure forward inference, so we inject stub modules that expose
    just the symbols the top-of-file ``from ... import`` lines need.
    """
    import types

    def _stub_missing(name: str) -> None:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    _stub_missing("create_DRR_AMED_original_pt_ver_mod")
    mod = sys.modules["create_DRR_AMED_original_pt_ver_mod"]
    mod.SPACING_DEFAULT = (2.0, 2.0, 2.0)
    def _euler_to_quaternion(*args, **kwargs):
        raise RuntimeError("euler_to_quaternion stub called — not available in inference-only build")
    mod.euler_to_quaternion = _euler_to_quaternion

    _stub_missing("DRR_norm_percentile")
    def _normalize_tensor_percentile_masked(*args, **kwargs):
        raise RuntimeError("normalize_tensor_percentile_masked stub called — not available in inference-only build")
    sys.modules["DRR_norm_percentile"].normalize_tensor_percentile_masked = _normalize_tensor_percentile_masked


def build_model(device: torch.device, *, C=96, n_heads=8, enc_base=48,
                unet_base=96, rho_bias_init=3.0, sr_base=32, sr_blocks=6,
                fusion_mode="attention", use_triplane_product=True,
                use_cross_attn=True, ca_heads=8, ckpt_path: Path = DEFAULT_CKPT):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    _install_dependency_stubs()
    from train_v2_attention_mod_multiL1_mod_perpectual2_original_large_model import (
        Stage0_2D_TriPlane, Stage1_Coarse3D, Stage2_SR3D, ModelMP3GPU,
    )

    triplane_mult = 4 if use_triplane_product else 3
    unet_in_ch = triplane_mult * C

    stage0 = Stage0_2D_TriPlane(C=C, n_heads=n_heads, enc_base=enc_base,
                                fusion_mode=fusion_mode,
                                use_triplane_product=use_triplane_product,
                                z_size=128, use_input_augmentation=False)
    stage1 = Stage1_Coarse3D(in_ch=unet_in_ch, unet_base=unet_base,
                             rho_bias_init=rho_bias_init,
                             use_cross_attn=use_cross_attn,
                             drr_ch=C, ca_heads=ca_heads)
    stage2 = Stage2_SR3D(sr_base=sr_base, sr_blocks=sr_blocks)
    dev = str(device)
    model = ModelMP3GPU(stage0, stage1, stage2, dev0=dev, dev1=dev, dev2=dev).eval()

    # The saved ckpt was trained against an older source where several
    # ``nn.Sequential`` blocks had no zero-dropout/Identity placeholder slot.
    # The current source adds them, which shifts later indices by one and
    # breaks strict load. Walk every nn.Sequential in the model and drop
    # any Identity child so the layer indices line up with the ckpt.
    import torch.nn as nn

    def _strip_identity_in_sequentials(root: nn.Module) -> int:
        replaced = 0
        for name, child in list(root.named_children()):
            if isinstance(child, nn.Sequential) and any(isinstance(m, nn.Identity) for m in child):
                kept = [m for m in child if not isinstance(m, nn.Identity)]
                setattr(root, name, nn.Sequential(*kept).to(device))
                replaced += 1
            replaced += _strip_identity_in_sequentials(getattr(root, name))
        return replaced

    n_patched = _strip_identity_in_sequentials(model)
    if n_patched:
        print(f"[ckpt compat] stripped Identity slots from {n_patched} Sequential blocks")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.stage0.load_state_dict(ckpt["stage0"], strict=True)
    model.stage1.load_state_dict(ckpt["stage1"], strict=True)
    model.stage2.load_state_dict(ckpt["stage2"], strict=True)
    print(f"[ckpt] loaded epoch={ckpt.get('epoch')}, step={ckpt.get('step')}")
    return model


@torch.no_grad()
def infer_frame(model, ap_2d: np.ndarray, lat_2d: np.ndarray, *,
                device: torch.device, alpha_delta: float, amp: bool, do_sr: bool) -> np.ndarray:
    ap = torch.from_numpy(ap_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    lat = torch.from_numpy(lat_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    use_amp = amp and device.type == "cuda"
    ct256, ct128, _ = model(ap, lat, alpha_delta=alpha_delta, amp=use_amp,
                            use_ckpt=False, do_sr=do_sr)
    out = ct256 if (do_sr and ct256 is not None) else ct128
    return out[0, 0].clamp(0, 1).detach().cpu().numpy().astype(np.float32)


def save_preview(volumes: np.ndarray, path: Path, *, case_id: str) -> None:
    """Pick a few phase frames, plot axial mid-slice as a row."""
    n = min(8, volumes.shape[0])
    idx = np.linspace(0, volumes.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3), squeeze=False)
    axes = axes.ravel()
    for ax, t in zip(axes, idx):
        ax.imshow(volumes[t, volumes.shape[1] // 2], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"case {case_id}: predicted 4DCT axial mid-slices")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_case(model, ap: np.ndarray, lat: np.ndarray, *, case_id: str,
             out_dir: Path, device: torch.device, alpha_delta: float,
             amp: bool, do_sr: bool) -> dict:
    case_out = out_dir / f"case_{case_id}"
    case_out.mkdir(parents=True, exist_ok=True)
    T = ap.shape[0]
    volumes: list[np.ndarray] = []
    timings: list[float] = []
    for t in range(T):
        t0 = time.time()
        ct = infer_frame(model, ap[t], lat[t], device=device,
                         alpha_delta=alpha_delta, amp=amp, do_sr=do_sr)
        timings.append(time.time() - t0)
        volumes.append(ct)
        if t % 4 == 0 or t == T - 1:
            print(f"    t={t:>3}/{T-1}  {timings[-1]:.2f}s  shape={ct.shape}  "
                  f"range=[{ct.min():.3f}, {ct.max():.3f}]")
    pred_4d = np.stack(volumes)  # (T, D, H, W)
    np.savez_compressed(case_out / "pred_4d_ct.npz",
                        ct=pred_4d, case_id=case_id)
    save_preview(pred_4d, case_out / "preview.png", case_id=case_id)

    metrics = {
        "case_id": case_id,
        "n_frames": int(T),
        "ct_shape": list(pred_4d.shape),
        "value_range": [float(pred_4d.min()), float(pred_4d.max())],
        "mean_inference_s_per_frame": float(np.mean(timings)),
        "total_inference_s": float(np.sum(timings)),
        "do_sr": bool(do_sr),
        "device": str(device),
    }
    (case_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {case_out / 'pred_4d_ct.npz'}  ({pred_4d.nbytes / 1e6:.1f} MB)")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        default=Path("outputs/lung4d_input/paired_dataset.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/lung4d_inference"))
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (auto-detected if unset)")
    parser.add_argument("--alpha-delta", type=float, default=0.8)
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP autocast (CUDA only)")
    parser.add_argument("--do-sr", action="store_true",
                        help="Run Stage2 SR128to256 (heavier, gives 256^3 instead of 128^3)")
    parser.add_argument("--cases", nargs="*",
                        help="Optional subset of case ids (e.g. 01 02)")
    parser.add_argument("--limit-frames", type=int,
                        help="Optional per-case frame cap (smoke testing)")
    args = parser.parse_args()

    if not args.bundle.is_file():
        print(f"bundle not found: {args.bundle}", file=sys.stderr); return 2
    device = pick_device(args.device)
    print(f"Using device: {device}")
    args.out.mkdir(parents=True, exist_ok=True)

    bundle = np.load(args.bundle, allow_pickle=True)
    ap_all = bundle["ap"]
    lat_all = bundle["lateral"]
    case_ids = list(bundle["case_ids"])
    print(f"Loaded bundle: AP={ap_all.shape}, LAT={lat_all.shape}, {len(case_ids)} cases")

    if args.cases:
        chosen = set(args.cases)
        keep_idx = [i for i, c in enumerate(case_ids) if c in chosen]
    else:
        keep_idx = list(range(len(case_ids)))
    if not keep_idx:
        print("No cases match filter", file=sys.stderr); return 2

    print("Building model ...")
    model = build_model(device=device, ckpt_path=args.ckpt)

    summary: list[dict] = []
    for i in keep_idx:
        case_id = case_ids[i]
        ap = ap_all[i]
        lat = lat_all[i]
        if args.limit_frames:
            ap = ap[: args.limit_frames]
            lat = lat[: args.limit_frames]
        print(f"\n=== case {case_id} ({ap.shape[0]} frames) ===")
        try:
            summary.append(run_case(model, ap, lat, case_id=case_id,
                                    out_dir=args.out, device=device,
                                    alpha_delta=args.alpha_delta,
                                    amp=args.amp, do_sr=args.do_sr))
        except Exception:
            import traceback; traceback.print_exc()

    if summary:
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        avg = sum(s["mean_inference_s_per_frame"] for s in summary) / len(summary)
        total = sum(s["total_inference_s"] for s in summary)
        print(f"\nDone. {len(summary)} cases, avg {avg:.2f}s/frame, "
              f"total {total/60:.1f} min")
        print(f"Outputs: {args.out}")
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
