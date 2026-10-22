"""One phase-5 W&B sweep trial: DINOv3-B n32 MIP, inner fold 0, from P3 stage 1.

W&B injects ``--unfreeze_encoder_blocks``, ``--learning_rate``, and
``--encoder_lr`` via ``${args}``. This script resolves the stage-1 ckpt and
the rest of the frozen P5 recipe, then execs ``main_train.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from run_plan import (
    DEFAULT_PLAN,
    TRAIN_SCRIPT,
    best_ckpt,
    expand_experiments,
    find_run,
    load_machine,
)

INIT_EXP = "P3_stage1_oldpenn_v3_b"
DATA_DIR = "final_cropped_and_masked_slabs_n32_s3_o0"
SPLIT_CSV = "new_penn_datasplit_v5_holdoutF0.csv"
FOLD = 0


def _hp_tag(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfreeze_encoder_blocks", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--encoder_lr", type=float, required=True)
    parser.add_argument("--machine", default="cluster")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    machine = load_machine(plan, args.machine)
    by_id = {e.id: e for e in expand_experiments(plan)}
    if INIT_EXP not in by_id:
        print(f"Plan is missing {INIT_EXP!r}.", file=sys.stderr)
        return 2
    run, done = find_run(machine, by_id[INIT_EXP], 0)
    if not done or run is None:
        print(f"Need a finished {INIT_EXP} fold 0 run under {machine.runs_root}.", file=sys.stderr)
        return 2
    ckpt = best_ckpt(run)

    data = machine.data_root / DATA_DIR
    if args.machine == "cluster":
        data = Path(str(data) + ".h5")
    split = machine.data_root / SPLIT_CSV
    lr_tag = _hp_tag(args.learning_rate)
    enc_tag = _hp_tag(args.encoder_lr)
    cohort = (
        f"P5_bayes_unfrz{args.unfreeze_encoder_blocks}_lr{lr_tag}_enc{enc_tag}_f{FOLD}"
    )
    workers = 8 if args.machine == "cluster" else 2

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset", "PENN",
        "--model_name", "DinoClassifierSliceV3",
        "--model_size", "b",
        "--use_registers", "true",
        "--penn_split_csv", str(split),
        "--path_root_data", str(data),
        "--slices", "32",
        "--slab_tissue_soft_weight",
        "--class_weight", "balanced",
        "--high-risk-policy", "malignant",
        "--dcis-policy", "malignant",
        "--fold", str(FOLD),
        "--seed", "123",
        "--batch_size", "4",
        "--num_workers", str(workers),
        "--path_root_output", str(machine.runs_root),
        "--ckpt_path", str(ckpt),
        "--cohort", cohort,
        "--unfreeze_encoder_blocks", str(args.unfreeze_encoder_blocks),
        "--learning_rate", str(args.learning_rate),
        "--encoder_lr", str(args.encoder_lr),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
