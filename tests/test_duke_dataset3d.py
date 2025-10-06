import os
from pathlib import Path

import pytest
import numpy as np

# Import the dataset class under test
from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D


@pytest.fixture(scope="module")
def duke_data_root():
    """
    Determine the data root for DUKE_Dataset3D.
    Uses the DUKE_Dataset3D.PATH_ROOT by default, but allows override via
    the DUKE_DATA_DIR environment variable for local testing.
    """
    env_override = os.getenv("DUKE_DATA_DIR")
    if env_override:
        return Path(env_override)
    return DUKE_Dataset3D.PATH_ROOT


@pytest.fixture(scope="module")
def duke_required_files(duke_data_root):
    """
    Return required files for this dataset to function.
    """
    return {
        "h5": duke_data_root / "data_compressed.h5",
        "csv": duke_data_root / "splits" / "split.csv",
    }


def test_duke_dataset3d_importable():
    # Simple import check
    assert DUKE_Dataset3D is not None


@pytest.mark.parametrize("fold", [0])
@pytest.mark.parametrize("split", ["train"])  # Adjust if your CSV uses different split naming
def test_duke_dataset3d_smoke(duke_data_root, duke_required_files, fold, split):
    # Verbose: show root and required files
    print(f"[DUKE] data_root={duke_data_root}")
    for k, p in duke_required_files.items():
        print(f"[DUKE] requires {k}: {p} (exists={p.exists()})")
    # Skip if required files are not present
    missing = [k for k, p in duke_required_files.items() if not p.exists()]
    if missing:
        pytest.skip(f"Skipping: missing required files for dataset: {missing}")

    # Attempt to create dataset using default transform
    ds = DUKE_Dataset3D(path_root=str(duke_data_root), fold=fold, split=split)
    # Verbose: show transform pipeline summary
    print(f"[DUKE] transform={ds.transform}")

    # Length can be zero depending on split; if zero, skip deep checks
    assert isinstance(len(ds), int)
    print(f"[DUKE] dataset length={len(ds)} for split='{split}', fold={fold}")
    if len(ds) == 0:
        pytest.skip("Split has zero samples; skipping item checks.")

    sample = ds[0]
    print(f"[DUKE] first uid={sample.get('uid')}")

    # Basic structure checks
    assert isinstance(sample, dict)
    for key in ["uid", "source", "target"]:
        assert key in sample, f"Missing key '{key}' in dataset sample"

    # Source should be a 3D tensor after default transforms
    source = sample["source"]
    try:
        import torch
    except Exception as e:
        pytest.skip(f"Torch unavailable in test environment: {e}")

    assert isinstance(source, torch.Tensor), "Expected 'source' to be a torch.Tensor after transforms"
    assert source.ndim == 4, f"Expected 4D tensor (C, X, Y, Z), got shape {tuple(source.shape)}"
    c, x, y, z = source.shape
    assert c in (1, 3), f"Unexpected channel count: {c}"
    # Ensure non-empty spatial dims
    assert x > 0 and y > 0 and z > 0
    # Verbose: source stats
    print(f"[DUKE] source shape={tuple(source.shape)} dtype={source.dtype} min={float(source.min()) if source.numel() else 'NA'} max={float(source.max()) if source.numel() else 'NA'} mean={float(source.float().mean()) if source.numel() else 'NA'}")

    # Target could be int/float/bool or numpy scalars; accept common scalar types
    target = sample["target"]
    acceptable_types = (int, float, bool, np.integer, np.floating, np.bool_, torch.Tensor)
    assert isinstance(target, acceptable_types), f"Unexpected target type: {type(target)}"
    if isinstance(target, torch.Tensor):
        assert target.numel() == 1, "Target tensor should be scalar-like"
        print(f"[DUKE] target (tensor) value={target.item()} dtype={target.dtype}")
    else:
        print(f"[DUKE] target value={target} type={type(target)}")
