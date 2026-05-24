"""Minimal example of consuming the lung4d_input bundle in a model.

Run with::

    python scripts/lung4d_loader_example.py --bundle outputs/lung4d_input/paired_dataset.npz

Demonstrates two access patterns:
1) Numpy-only iteration (for inspection / non-PyTorch code).
2) PyTorch DataLoader wrapper (for training/inference loops).

The bundle stores AP and LAT as ``float32`` ``(N, T, H, W)`` in ``[0, 1]``.
Cast / re-normalise as your model expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_bundle(path: Path):
    bundle = np.load(path, allow_pickle=True)
    return {
        "ap": bundle["ap"],            # (N, T, H, W) float32 [0, 1]
        "lateral": bundle["lateral"],  # (N, T, H, W) float32 [0, 1]
        "case_ids": list(bundle["case_ids"]),
    }


def iterate_numpy(bundle: dict):
    for i, case_id in enumerate(bundle["case_ids"]):
        ap = bundle["ap"][i]      # (T, H, W)
        lat = bundle["lateral"][i]  # (T, H, W)
        yield case_id, ap, lat


def make_torch_dataset(bundle: dict):
    """Lazy import torch; returns a Dataset wrapping the bundle."""
    import torch
    from torch.utils.data import Dataset

    class PairedCycleDataset(Dataset):
        def __init__(self, ap, lateral, case_ids):
            self.ap = torch.from_numpy(ap).float()           # (N, T, H, W)
            self.lateral = torch.from_numpy(lateral).float()
            self.case_ids = list(case_ids)

        def __len__(self):
            return len(self.case_ids)

        def __getitem__(self, idx):
            return {
                "case_id": self.case_ids[idx],
                "ap": self.ap[idx],            # (T, H, W)
                "lateral": self.lateral[idx],  # (T, H, W)
            }

    return PairedCycleDataset(bundle["ap"], bundle["lateral"], bundle["case_ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    print("Bundle keys:", list(bundle))
    print(f"AP:      {bundle['ap'].shape}, dtype={bundle['ap'].dtype}, "
          f"range=[{bundle['ap'].min():.3f}, {bundle['ap'].max():.3f}]")
    print(f"Lateral: {bundle['lateral'].shape}, dtype={bundle['lateral'].dtype}, "
          f"range=[{bundle['lateral'].min():.3f}, {bundle['lateral'].max():.3f}]")
    print(f"Case IDs ({len(bundle['case_ids'])}): {bundle['case_ids']}")
    print()

    print("Numpy iteration sample:")
    for i, (case_id, ap, lat) in enumerate(iterate_numpy(bundle)):
        print(f"  case {case_id}: AP {ap.shape}  LAT {lat.shape}")
        if i >= 2:
            print(f"  ... ({len(bundle['case_ids']) - 3} more cases)")
            break

    try:
        print("\nPyTorch Dataset sample:")
        ds = make_torch_dataset(bundle)
        sample = ds[0]
        print(f"  __getitem__(0): case_id={sample['case_id']}, "
              f"ap={tuple(sample['ap'].shape)}, lateral={tuple(sample['lateral'].shape)}")
    except ImportError:
        print("\n(torch not installed; skipping PyTorch demo)")


if __name__ == "__main__":
    main()
