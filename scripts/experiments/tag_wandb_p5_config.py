"""Write unfreeze/LR keys onto existing P5 W&B runs so a Bayesian sweep can seed.

Lightning did not log ``unfreeze_encoder_blocks`` / ``learning_rate`` as
top-level config. This copies them from the run name (plan cohort tag) onto
``run.config``, which is what the W&B GP reads.

    python scripts/experiments/tag_wandb_p5_config.py --entity <entity>
    python scripts/experiments/tag_wandb_p5_config.py --entity <entity> --dry-run
"""

from __future__ import annotations

import argparse
import re

import wandb

PROJECT = "Classifier_PENN_MST"
# P5_staged_v3_b_unfrz2_lr1e-05_enc1e-06_f0  (and P5_bayes_...)
HP_RE = re.compile(
    r"unfrz(?P<unfrz>\d+)_lr(?P<lr>[0-9.eE+-]+)_enc(?P<enc>[0-9.eE+-]+)",
)
BASELINE = {
    "P3_staged_v3_b": {
        "unfreeze_encoder_blocks": 2,
        "learning_rate": 1e-5,
        "encoder_lr": 1e-6,
    },
}
KEYS = ("unfreeze_encoder_blocks", "learning_rate", "encoder_lr")


def hps_from_name(name: str) -> dict | None:
    m = HP_RE.search(name or "")
    if m:
        return {
            "unfreeze_encoder_blocks": int(m.group("unfrz")),
            "learning_rate": float(m.group("lr")),
            "encoder_lr": float(m.group("enc")),
        }
    for tag, hps in BASELINE.items():
        if tag in (name or ""):
            return dict(hps)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    tagged = already = 0
    for run in api.runs(path):
        name = run.name or ""
        if not any(tag in name for tag in ("P5_staged_v3", "P5_bayes", "P3_staged_v3_b")):
            continue
        if "_f0" not in name and (run.config or {}).get("fold") not in (0, "0"):
            continue
        hps = hps_from_name(name)
        if not hps:
            continue
        cfg = dict(run.config or {})
        if all(cfg.get(k) is not None for k in KEYS):
            already += 1
            continue
        print(f"{run.id}  {name}  +{hps}")
        tagged += 1
        if args.dry_run:
            continue
        run.config.update(hps)
        run.update()
    print(f"updated={tagged} already_ok={already} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
