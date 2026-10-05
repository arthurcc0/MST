"""Drive the PENN experiment plan: train + predict per fold, then summarize CV AUC.

The plan (``experiments/plan.yaml``) is the only file you edit. Status is
derived from disk every time (a run is *trained* when its run folder has
``best_checkpoint.json``; *evaluated* when ``results_<split>.csv`` exists), so
re-running a command skips finished folds and resumes after a crash or on a
different machine once ``runs/`` and the predict results are copied over.

    python scripts/experiments/run_plan.py list     [--phase 1]
    python scripts/experiments/run_plan.py status   [--phase 1]
    python scripts/experiments/run_plan.py run      --phase 1 [--exp ID ...] [--folds 0 1] [--dry-run]
    python scripts/experiments/run_plan.py summarize
    python scripts/experiments/run_plan.py times       # rebuild training_times.csv from runs + ledger
    python scripts/experiments/run_plan.py orphans      # run folders the plan does not use

Outputs (next to the plan): ``summary.csv`` / ``summary.xlsx`` (one row per
experiment, fold and mean AUCs), ``ledger.csv`` (append-only log of every
launched command), ``results/training_times.csv``, ``logs/<exp>/``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
from aggregate_auc import results_dir_name  # noqa: E402

DEFAULT_PLAN = PROJECT_ROOT / "experiments" / "plan.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "main_train.py"
PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "predict" / "main_predict.py"
# main_predict.py writes under <output_dir>/<this>/<dataset>/<results_dir_name>.
PREDICT_RESULTS_FOLDER = "results-pretrained-oldpenn-on-newpenn"

STORE_TRUE_FLAGS = {
    "freeze_backbone",
    "slab_tissue_soft_weight",
    "only_malignants",
    "with_laterality",
    "use_clinical_notes",
}
VALUE_BOOL_FLAGS = {"use_registers", "paired_sampling"}
FLAG_SPELLING = {
    "high_risk_policy": "--high-risk-policy",
    "dcis_policy": "--dcis-policy",
    "focal_gamma": "--focal-gamma",
    "focal_alpha": "--focal-alpha",
    "results_name": "--results-name",
}
RUNNER_ONLY_KEYS = {"batch_size_by_model_size"}
SLAB_KEYS = ("slab_tissue_soft_weight", "slab_tissue_min_weight", "slab_tissue_keep_ratio")
SUMMARY_PARAM_KEYS = (
    "penn_split_csv",
    "path_root_data",
    "model_size",
    "slices",
    "freeze_backbone",
    "unfreeze_encoder_blocks",
    "learning_rate",
    "encoder_lr",
    "class_weight",
    "loss",
    "high_risk_policy",
    "seed",
    "batch_size",
)


class _KeepMissing(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _fmt(value, params: dict):
    if isinstance(value, str):
        return value.format_map(_KeepMissing(params))
    if isinstance(value, dict):
        return {k: _fmt(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_fmt(v, params) for v in value]
    return value


@dataclass
class Experiment:
    id: str
    phase: int
    description: str
    enabled: bool
    cohort: str
    folds: list[int]
    train: dict
    eval: dict | None
    init_from: dict | None
    eval_from: dict | None
    note: str = ""
    grid_values: dict = field(default_factory=dict)

    def cohort_for(self, fold: int) -> str:
        return f"{self.cohort}_f{fold}"


@dataclass
class Machine:
    name: str
    python: str
    data_root: Path
    runs_root: Path
    predict_output_dir: Path
    # Applied on top of each experiment's train settings on this machine only.
    # Values may reference the experiment's own settings, e.g.
    # path_root_data: "{path_root_data}.h5". Run names and summary.csv use
    # the plan values, so the same experiment matches across machines.
    train_overrides: dict = field(default_factory=dict)
    # Extra main_predict flags on this machine (e.g. a data-format switch).
    predict_overrides: dict = field(default_factory=dict)

    @property
    def results_root(self) -> Path:
        return self.predict_output_dir / PREDICT_RESULTS_FOLDER / "PENN"


# ---------------------------------------------------------------- plan loading


def _abs(path: str | Path, base: Path) -> Path:
    p = Path(os.path.expandvars(str(path)))
    return p if p.is_absolute() else (base / p)


def load_machine(plan: dict, name: str | None) -> Machine:
    machines = plan.get("machines") or {}
    key = name or socket.gethostname()
    if key not in machines:
        if name:
            raise KeyError(f"Machine {name!r} not in plan machines {sorted(machines)}")
        key = "default"
    cfg = machines.get(key) or {}
    return Machine(
        name=key,
        python=cfg.get("python") or sys.executable,
        data_root=_abs(cfg.get("data_root", "."), PROJECT_ROOT),
        runs_root=_abs(cfg.get("runs_root", "runs"), PROJECT_ROOT),
        predict_output_dir=_abs(cfg.get("predict_output_dir", "."), PROJECT_ROOT),
        train_overrides=dict(cfg.get("train_overrides") or {}),
        predict_overrides=dict(cfg.get("predict_overrides") or {}),
    )


def expand_experiments(plan: dict) -> list[Experiment]:
    defaults_train = dict(plan.get("defaults", {}).get("train", {}))
    defaults_eval = dict(plan.get("defaults", {}).get("eval", {}))
    default_folds = list(plan.get("defaults", {}).get("folds", [0, 1, 2, 3, 4]))
    out: list[Experiment] = []
    seen: set[str] = set()
    for spec in plan.get("experiments", []):
        grid = spec.get("grid") or {}
        # vars expand like grid but only fill {templates}; they are not train args.
        template_vars = spec.get("vars") or {}
        axes = {**grid, **template_vars}
        keys = list(axes)
        combos = list(itertools.product(*(axes[k] for k in keys))) or [()]
        for combo in combos:
            values = dict(zip(keys, combo))
            grid_values = {k: v for k, v in values.items() if k in grid}
            train = {**defaults_train, **(spec.get("train") or {}), **grid_values}
            params = {**train, **values}
            exp_id = _fmt(spec["id"], params)
            if exp_id in seen:
                raise ValueError(f"Duplicate experiment id {exp_id!r}; add grid fields to the id template.")
            seen.add(exp_id)
            params["id"] = exp_id
            train = _fmt(train, params)
            eval_spec = spec.get("eval", {})
            eval_cfg = None if eval_spec is False else {**defaults_eval, **(eval_spec or {})}
            init_from = _fmt(spec.get("init_from"), params) if spec.get("init_from") else None
            eval_from = _fmt(spec.get("eval_from"), params) if spec.get("eval_from") else None
            out.append(
                Experiment(
                    id=exp_id,
                    phase=int(spec.get("phase", 0)),
                    description=_fmt(spec.get("description", ""), params),
                    enabled=bool(spec.get("enabled", True)),
                    cohort=_fmt(spec.get("cohort", exp_id), params),
                    folds=list(spec.get("folds", default_folds)),
                    train=train,
                    eval=eval_cfg,
                    init_from=init_from,
                    eval_from=eval_from,
                    note=_fmt(spec.get("note", ""), params),
                    grid_values=values,
                )
            )
    return out


def resolved_train_args(exp: Experiment) -> dict:
    args = {k: v for k, v in exp.train.items() if k not in RUNNER_ONLY_KEYS}
    by_size = exp.train.get("batch_size_by_model_size") or {}
    if "batch_size" not in args and by_size:
        size = str(args.get("model_size", "s"))
        if size in by_size:
            args["batch_size"] = by_size[size]
    return args


def machine_train_args(machine: Machine, exp: Experiment) -> dict:
    """Train args as run on ``machine``: plan settings plus machine overrides."""
    args = resolved_train_args(exp)
    params = {**exp.grid_values, **args}
    args.update(_fmt(machine.train_overrides, params))
    return args


# ---------------------------------------------------------------- disk state


def _run_folder_regex(input_type: str, cohort_f: str) -> re.Pattern:
    tags = r"(?:_v3)?(?:_vit[slbg])?(?:_reg)?(?:_d\d+)?(?:_multi)?"
    return re.compile(rf"_{re.escape(input_type)}_{re.escape(cohort_f)}{tags}$")


def matching_runs(machine: Machine, exp: Experiment, fold: int) -> list[Path]:
    root = machine.runs_root / str(exp.train.get("dataset", "PENN"))
    if not root.is_dir():
        return []
    pattern = _run_folder_regex(str(exp.train.get("input_type", "subtraction")), exp.cohort_for(fold))
    return [p for p in root.iterdir() if p.is_dir() and pattern.search(p.name)]


def find_run(machine: Machine, exp: Experiment, fold: int) -> tuple[Path | None, bool]:
    """Newest run folder for this fold, and whether it finished training."""
    matches = matching_runs(machine, exp, fold)
    if not matches:
        return None, False
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in matches:
        if (p / "best_checkpoint.json").is_file():
            return p, True
    return matches[0], False


def best_ckpt(run: Path) -> Path:
    with open(run / "best_checkpoint.json", encoding="utf-8") as f:
        return run / json.load(f)["best_model_epoch"]


def results_csv(machine: Machine, exp: Experiment, run: Path, fold: int, split: str) -> Path:
    name = exp.cohort_for(fold) if exp.eval_from else results_dir_name(run, fold=fold)
    return machine.results_root / name / f"results_{split}.csv"


def _data_path(machine: Machine, value) -> Path:
    return _abs(value, machine.data_root)


def preflight(machine: Machine, exp: Experiment) -> list[str]:
    """Reasons this experiment cannot start yet (missing inputs)."""
    problems = []
    args = machine_train_args(machine, exp)
    for key in ("penn_split_csv", "path_root_data"):
        if args.get(key) and not _data_path(machine, args[key]).exists():
            problems.append(f"missing {key}: {_data_path(machine, args[key])}")
    if args.get("ckpt_path") and not _abs(args["ckpt_path"], PROJECT_ROOT).is_file():
        problems.append(f"missing ckpt_path: {args['ckpt_path']}")
    return problems


def resolve_source_run(
    machine: Machine, exps: dict[str, Experiment], spec: dict | None, fold: int, kind: str,
) -> tuple[Path | None, str]:
    if not spec:
        return None, ""
    src_id = spec["experiment"]
    if src_id not in exps:
        return None, f"{kind} experiment {src_id!r} is not in the plan"
    src_fold = spec.get("fold", "same")
    src_fold = fold if str(src_fold) == "same" else int(src_fold)
    run, done = find_run(machine, exps[src_id], src_fold)
    if not done:
        return None, f"waiting for {src_id} fold {src_fold} to finish training"
    return run, ""


def resolve_init_ckpt(machine: Machine, exps: dict[str, Experiment], exp: Experiment, fold: int) -> tuple[Path | None, str]:
    run, wait = resolve_source_run(machine, exps, exp.init_from, fold, "init_from")
    if wait or run is None:
        return None, wait
    return best_ckpt(run), ""


# ---------------------------------------------------------------- commands


def _flag_args(args: dict) -> list[str]:
    cmd: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = FLAG_SPELLING.get(key, f"--{key}")
        if key in STORE_TRUE_FLAGS:
            if value:
                cmd.append(flag)
        elif key in VALUE_BOOL_FLAGS:
            cmd += [flag, "true" if value else "false"]
        else:
            cmd += [flag, str(value)]
    return cmd


def train_command(machine: Machine, exp: Experiment, fold: int, init_ckpt: Path | None) -> list[str]:
    args = machine_train_args(machine, exp)
    args["fold"] = fold
    args["cohort"] = exp.cohort_for(fold)
    args["path_root_output"] = str(machine.runs_root)
    for key in ("penn_split_csv", "path_root_data"):
        if args.get(key):
            args[key] = str(_data_path(machine, args[key]))
    if init_ckpt is not None:
        args["ckpt_path"] = str(init_ckpt)
    elif args.get("ckpt_path"):
        args["ckpt_path"] = str(_abs(args["ckpt_path"], PROJECT_ROOT))

    return [machine.python, str(TRAIN_SCRIPT), *_flag_args(args)]


def predict_command(machine: Machine, exp: Experiment, run: Path, fold: int, split: str) -> list[str]:
    cmd = [
        machine.python,
        str(PREDICT_SCRIPT),
        "--run_dir", str(run.parent.parent),
        "--run_folder", f"{run.parent.name}/{run.name}",
        "--fold", str(fold),
        "--split", split,
        "--output_dir", str(machine.predict_output_dir),
    ]
    # config.yaml stores the training machine's absolute paths; resolve for this one.
    args = machine_train_args(machine, exp)
    for key in ("path_root_data", "penn_split_csv"):
        if args.get(key):
            cmd += [f"--{key}", str(_data_path(machine, args[key]))]
    ev = exp.eval or {}
    if ev.get("high_risk_policy"):
        cmd += ["--high-risk-policy", ev["high_risk_policy"]]
    if ev.get("dcis_policy"):
        cmd += ["--dcis-policy", ev["dcis_policy"]]
    # Slab weighting is not stored in config.yaml by older runs; mirror training.
    if exp.train.get("slab_tissue_soft_weight"):
        cmd.append("--slab_tissue_soft_weight")
        for key in SLAB_KEYS[1:]:
            if exp.train.get(key) is not None:
                cmd += [f"--{key}", str(exp.train[key])]
    if exp.eval_from:
        cmd += ["--results-name", exp.cohort_for(fold)]
    params = {**exp.grid_values, **args}
    cmd += _flag_args(_fmt(machine.predict_overrides, params))
    return cmd


def _git_rev() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(subprocess.list2cmdline(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def _append_ledger(path: Path, row: dict) -> None:
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


TIMES_FIELDS = [
    "experiment", "fold", "model", "gpu", "epochs", "best_epoch",
    "start_time", "end_time", "training_seconds", "training_minutes", "source",
]


def _epoch_from_ckpt_name(name: str) -> int | None:
    m = re.search(r"epoch[=-](\d+)", name)
    return int(m.group(1)) if m else None


def _epochs_from_run(run: Path) -> tuple[int | None, int | None, str]:
    """Return (epochs_ran, best_epoch, gpu) from a finished run folder."""
    stats_path = run / "training_stats.json"
    if stats_path.is_file():
        d = json.loads(stats_path.read_text(encoding="utf-8"))
        return d.get("epochs_ran"), d.get("best_epoch"), d.get("gpu") or ""
    best_epoch = None
    ckpt_json = run / "best_checkpoint.json"
    if ckpt_json.is_file():
        name = json.loads(ckpt_json.read_text(encoding="utf-8")).get("best_model_epoch", "")
        best_epoch = _epoch_from_ckpt_name(str(name))
    last_epoch = None
    last = run / "last.ckpt"
    if last.is_file():
        try:
            import torch
            ckpt = torch.load(last, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and ckpt.get("epoch") is not None:
                last_epoch = int(ckpt["epoch"])
        except Exception:
            last_epoch = None
    epochs_ran = (last_epoch + 1) if last_epoch is not None else None
    return epochs_ran, best_epoch, ""


def _span_from_run_files(run: Path) -> tuple[datetime | None, datetime | None, float | None]:
    """Wall clock from config.yaml (written at train start) to best_checkpoint.json."""
    end_p = run / "best_checkpoint.json"
    start_p = run / "config.yaml"
    if not end_p.is_file():
        return None, None, None
    if not start_p.is_file():
        start_p = min((p for p in run.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime, default=end_p)
    start = datetime.fromtimestamp(start_p.stat().st_mtime)
    end = datetime.fromtimestamp(end_p.stat().st_mtime)
    seconds = (end - start).total_seconds()
    if seconds < 0:
        return start, end, None
    return start, end, seconds


def _ledger_train_times(plan_dir: Path) -> dict[tuple[str, int], dict]:
    path = plan_dir / "ledger.csv"
    out: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("step") != "train" or str(row.get("returncode", "0")) not in {"0", "0.0"}:
                continue
            try:
                key = (row["experiment"], int(row["fold"]))
            except (KeyError, ValueError):
                continue
            out[key] = row
    return out


def _write_times_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMES_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TIMES_FIELDS})


def _append_training_times(plan_dir: Path, exp: Experiment, fold: int, run: Path | None,
                           start: datetime, seconds: float, gpu: str) -> None:
    epochs, best_epoch, stats_gpu = (None, None, "")
    if run is not None:
        epochs, best_epoch, stats_gpu = _epochs_from_run(run)
    path = plan_dir / "results" / "training_times.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMES_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({
            "experiment": exp.id,
            "fold": fold,
            "model": exp.train.get("model_name", ""),
            "gpu": stats_gpu or gpu,
            "epochs": epochs if epochs is not None else "",
            "best_epoch": best_epoch if best_epoch is not None else "",
            "start_time": start.isoformat(timespec="seconds"),
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "training_seconds": round(seconds, 2),
            "training_minutes": round(seconds / 60, 2),
            "source": "runner",
        })


def cmd_times(machine: Machine, exps: list[Experiment], plan_dir: Path) -> None:
    """Rebuild training_times.csv from finished run folders and ledger.csv."""
    ledger = _ledger_train_times(plan_dir)
    rows = []
    for exp in exps:
        for fold in exp.folds:
            run, done = find_run(machine, exp, fold)
            if not done or run is None:
                continue
            epochs, best_epoch, gpu = _epochs_from_run(run)
            start, end, file_seconds = _span_from_run_files(run)
            led = ledger.get((exp.id, fold))
            source = "files"
            seconds = file_seconds
            if led and led.get("minutes") not in (None, ""):
                try:
                    seconds = float(led["minutes"]) * 60
                    source = "ledger"
                    stamp = led.get("timestamp") or ""
                    if stamp:
                        start = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
                        end = start + timedelta(seconds=seconds)
                except (TypeError, ValueError):
                    pass
            rows.append({
                "experiment": exp.id,
                "fold": fold,
                "model": exp.train.get("model_name", ""),
                "gpu": gpu,
                "epochs": epochs if epochs is not None else "",
                "best_epoch": best_epoch if best_epoch is not None else "",
                "start_time": start.isoformat(timespec="seconds") if start else "",
                "end_time": end.isoformat(timespec="seconds") if end else "",
                "training_seconds": round(seconds, 2) if seconds is not None else "",
                "training_minutes": round(seconds / 60, 2) if seconds is not None else "",
                "source": source,
            })
    out = plan_dir / "results" / "training_times.csv"
    _write_times_csv(out, rows)
    print(f"Wrote {len(rows)} rows to {out}")


def execute(step: str, cmd: list[str], exp: Experiment, fold: int, machine: Machine,
            plan_dir: Path, dry_run: bool) -> bool:
    print(f"\n[{exp.id} f{fold}] {step}:\n  {subprocess.list2cmdline(cmd)}", flush=True)
    if dry_run:
        return True
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = plan_dir / "logs" / exp.id / f"f{fold}_{step}_{stamp}.log"
    start = datetime.now()
    t0 = time.perf_counter()
    code = _run_logged(cmd, log_path)
    elapsed = time.perf_counter() - t0
    _append_ledger(plan_dir / "ledger.csv", {
        "timestamp": stamp,
        "machine": machine.name,
        "host": socket.gethostname(),
        "git": _git_rev(),
        "experiment": exp.id,
        "phase": exp.phase,
        "fold": fold,
        "step": step,
        "returncode": code,
        "minutes": round(elapsed / 60, 1),
        "log": str(log_path.relative_to(plan_dir)),
        "command": subprocess.list2cmdline(cmd),
    })
    if step == "train":
        run, _ = find_run(machine, exp, fold)
        gpu = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = os.environ.get("SLURM_JOB_PARTITION", "")
        _append_training_times(plan_dir, exp, fold, run, start, elapsed, gpu)
    if code != 0:
        print(f"[{exp.id} f{fold}] {step} FAILED (exit {code}); log: {log_path}", flush=True)
    return code == 0


# ---------------------------------------------------------------- status / summary


def fold_state(machine: Machine, exp: Experiment, fold: int,
               exps: dict[str, Experiment] | None = None) -> dict:
    if exp.eval_from and exps is not None:
        run, wait = resolve_source_run(machine, exps, exp.eval_from, fold, "eval_from")
        trained = run is not None and not wait
    else:
        run, trained = find_run(machine, exp, fold)
    state = {"run": run, "trained": trained, "aucs": {}, "n": {}}
    if run is None or not trained or not exp.eval:
        return state
    for split in exp.eval.get("splits", ["test"]):
        path = results_csv(machine, exp, run, fold, split)
        if path.is_file():
            df = pd.read_csv(path)
            state["aucs"][split] = float(roc_auc_score(df["GT"], df["NN_pred"]))
            state["n"][split] = len(df)
            state.setdefault("frames", {})[split] = df
    return state


def experiment_row(machine: Machine, exp: Experiment, exps: dict[str, Experiment]) -> dict:
    states = {f: fold_state(machine, exp, f, exps) for f in exp.folds}
    splits = (exp.eval or {}).get("splits", [])
    n_trained = sum(s["trained"] for s in states.values())
    n_eval = sum(all(sp in s["aucs"] for sp in splits) for s in states.values()) if splits else 0
    blocked = preflight(machine, exp)
    if exp.init_from and n_trained < len(exp.folds):
        waits = {resolve_init_ckpt(machine, exps, exp, f)[1] for f in exp.folds}
        blocked += sorted(w for w in waits if w)
    if exp.eval_from and n_trained < len(exp.folds):
        waits = {resolve_source_run(machine, exps, exp.eval_from, f, "eval_from")[1] for f in exp.folds}
        blocked += sorted(w for w in waits if w)
    if not exp.enabled:
        status = "disabled"
    elif splits and n_eval == len(exp.folds):
        status = "done"
    elif not splits and n_trained == len(exp.folds):
        status = "done"
    elif blocked:
        status = "blocked"
    elif n_trained or any(s["run"] for s in states.values()):
        status = "partial"
    else:
        status = "todo"
    if status == "done":
        blocked = []  # e.g. trained elsewhere; local data paths no longer matter
    args = resolved_train_args(exp)
    row = {
        "phase": exp.phase,
        "experiment": exp.id,
        "status": status,
        "folds_trained": f"{n_trained}/{len(exp.folds)}",
        "folds_evaluated": f"{n_eval}/{len(exp.folds)}" if splits else "-",
    }
    for split in splits:
        aucs = [s["aucs"][split] for s in states.values() if split in s["aucs"]]
        row[f"{split}_auc_mean"] = round(float(pd.Series(aucs).mean()), 4) if aucs else None
        row[f"{split}_auc_std"] = round(float(pd.Series(aucs).std(ddof=1)), 4) if len(aucs) > 1 else None
        frames = [s["frames"][split] for s in states.values() if split in s.get("frames", {})]
        if len(frames) == len(exp.folds):
            pool = pd.concat(frames, ignore_index=True)
            row[f"{split}_auc_pooled"] = round(float(roc_auc_score(pool["GT"], pool["NN_pred"])), 4)
        for f, s in states.items():
            if split in s["aucs"]:
                row[f"{split}_f{f}"] = round(s["aucs"][split], 4)
    row["init_from"] = (
        exp.init_from["experiment"] if exp.init_from
        else (f"eval {exp.eval_from['experiment']}" if exp.eval_from else "")
    )
    for key in SUMMARY_PARAM_KEYS:
        row[key] = args.get(key)
    row["cohort"] = exp.cohort
    row["description"] = exp.description
    row["note"] = exp.note
    row["blocked_by"] = "; ".join(blocked)
    return row


def write_summary(machine: Machine, exps: list[Experiment], plan_dir: Path) -> pd.DataFrame:
    by_id = {e.id: e for e in exps}
    rows = [experiment_row(machine, e, by_id) for e in exps]
    df = pd.DataFrame(rows)
    lead = [c for c in ("phase", "experiment", "status", "folds_trained", "folds_evaluated",
                        "val_auc_mean", "val_auc_std", "test_auc_mean", "test_auc_std",
                        "test_auc_pooled") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(plan_dir / "summary.csv", index=False)
    try:
        df.to_excel(plan_dir / "summary.xlsx", index=False)
    except (ImportError, PermissionError) as e:
        print(f"(summary.xlsx not written: {e})")
    return df


def print_status(df: pd.DataFrame) -> None:
    cols = [c for c in ("phase", "experiment", "status", "folds_trained", "folds_evaluated",
                        "val_auc_mean", "test_auc_mean", "test_auc_std", "blocked_by")
            if c in df.columns]
    view = df[cols].copy()
    if "blocked_by" in view.columns:
        view["blocked_by"] = view["blocked_by"].fillna("").map(
            lambda s: s if len(s) <= 60 else s[:57] + "..."
        )
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(view.to_string(index=False))


def _folder_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def cmd_orphans(machine: Machine, exps: list[Experiment], plan_dir: Path) -> None:
    """List run folders the plan does not use (for archiving / deleting by hand)."""
    used: dict[Path, str] = {}
    for exp in exps:
        for fold in exp.folds:
            runs = sorted(matching_runs(machine, exp, fold), key=lambda p: p.stat().st_mtime, reverse=True)
            newest_done = next((p for p in runs if (p / "best_checkpoint.json").is_file()), None)
            for p in runs:
                used[p] = exp.id if p == newest_done else f"{exp.id} (older duplicate)"
    for run in list(used):
        cfg = run / "config.yaml"
        if cfg.is_file():
            init = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("ckpt_init")
            if init:
                parent = _abs(init, PROJECT_ROOT).parent
                used.setdefault(parent, f"stage-1 ckpt of {used[run].split(' ')[0]}")

    rows = []
    root = machine.runs_root
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for run in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            key = next((u for u in used if u.resolve() == run.resolve()), None)
            use = used.get(key, "")
            if use and "duplicate" not in use:
                continue
            rows.append({
                "run": f"{ds_dir.name}/{run.name}",
                "reason": use or "not in plan",
                "finished": (run / "best_checkpoint.json").is_file(),
                "gb": round(_folder_gb(run), 2),
            })
    df = pd.DataFrame(rows, columns=["run", "reason", "finished", "gb"])
    out = plan_dir / "orphans.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 140):
        print(df.to_string(index=False))
    print(f"\n{len(df)} folders, {df['gb'].sum():.1f} GB not used by the plan. Listed in {out}")


# ---------------------------------------------------------------- main


def select(exps: list[Experiment], phases, ids) -> list[Experiment]:
    out = exps
    if phases:
        out = [e for e in out if e.phase in set(phases)]
    if ids:
        wanted = set(ids)
        unknown = wanted - {e.id for e in exps}
        if unknown:
            raise SystemExit(f"Unknown experiment id(s): {sorted(unknown)}")
        out = [e for e in out if e.id in wanted]
    return out


def cmd_run(args, machine: Machine, exps: list[Experiment], plan_dir: Path) -> None:
    by_id = {e.id: e for e in exps}
    chosen = select(exps, args.phase, args.exp)
    launched = 0
    for exp in chosen:
        if not exp.enabled and not args.exp:
            print(f"[{exp.id}] disabled in plan; skipping (name it with --exp to force).")
            continue
        problems = preflight(machine, exp)
        if problems:
            print(f"[{exp.id}] blocked: {'; '.join(problems)}")
            continue
        folds = [f for f in exp.folds if not args.folds or f in args.folds]
        for fold in folds:
            if args.max_runs is not None and launched >= args.max_runs:
                print(f"Reached --max-runs {args.max_runs}.")
                return
            if exp.eval_from:
                run, wait = resolve_source_run(machine, by_id, exp.eval_from, fold, "eval_from")
                if wait:
                    print(f"[{exp.id} f{fold}] blocked: {wait}")
                    continue
                trained = True
            else:
                run, trained = find_run(machine, exp, fold)
            if not trained and not args.eval_only and not exp.eval_from:
                ckpt, wait = resolve_init_ckpt(machine, by_id, exp, fold)
                if wait:
                    print(f"[{exp.id} f{fold}] blocked: {wait}")
                    continue
                if run is not None and not args.restart_unfinished:
                    print(f"[{exp.id} f{fold}] unfinished run folder {run.name} (still running or crashed); "
                          "skipping. Pass --restart-unfinished to train it again.")
                    continue
                ok = execute("train", train_command(machine, exp, fold, ckpt), exp, fold,
                             machine, plan_dir, args.dry_run)
                launched += 1
                if not ok:
                    continue
                if args.dry_run:
                    continue
                run, trained = find_run(machine, exp, fold)
                if not trained:
                    print(f"[{exp.id} f{fold}] training exited but no best_checkpoint.json found.")
                    continue
            if not trained or args.skip_eval or not exp.eval:
                continue
            for split in exp.eval.get("splits", ["test"]):
                if results_csv(machine, exp, run, fold, split).is_file():
                    continue
                execute(f"predict_{split}", predict_command(machine, exp, run, fold, split),
                        exp, fold, machine, plan_dir, args.dry_run)
    if not args.dry_run:
        print_status(write_summary(machine, exps, plan_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "status", "run", "summarize", "times", "orphans"])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--machine", default=None, help="Key under machines: in the plan (default: hostname, else 'default').")
    parser.add_argument("--phase", type=int, nargs="+", default=None)
    parser.add_argument("--exp", nargs="+", default=None, help="Experiment id(s) after grid expansion.")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Only these folds (e.g. screen on 0 1).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-eval", action="store_true", help="Train only; predict later.")
    parser.add_argument("--eval-only", action="store_true", help="Predict finished runs; never train.")
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after launching N trainings.")
    parser.add_argument("--restart-unfinished", action="store_true",
                        help="Retrain folds whose newest run folder has no best_checkpoint.json.")
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    with open(plan_path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    machine = load_machine(plan, args.machine)
    exps = expand_experiments(plan)
    plan_dir = plan_path.parent
    print(f"Plan: {plan_path}  machine: {machine.name}  runs: {machine.runs_root}  results: {machine.results_root}")

    if args.command == "list":
        for e in select(exps, args.phase, args.exp):
            flag = "" if e.enabled else "  [disabled]"
            print(f"P{e.phase}  {e.id}  folds={e.folds}{flag}\n      {e.description}")
            for fold in e.folds[:1]:
                ckpt_note = f" + ckpt from {e.init_from['experiment']}" if e.init_from else ""
                print(f"      {subprocess.list2cmdline(train_command(machine, e, fold, None))}{ckpt_note}")
    elif args.command in {"status", "summarize"}:
        df = write_summary(machine, exps, plan_dir)
        if args.phase or args.exp:
            ids = {e.id for e in select(exps, args.phase, args.exp)}
            df = df[df["experiment"].isin(ids)]
        print_status(df)
        print(f"\nWrote {plan_dir / 'summary.csv'}")
    elif args.command == "times":
        cmd_times(machine, select(exps, args.phase, args.exp), plan_dir)
    elif args.command == "orphans":
        cmd_orphans(machine, exps, plan_dir)
    else:
        cmd_run(args, machine, exps, plan_dir)


if __name__ == "__main__":
    main()
