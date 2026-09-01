#!/usr/bin/env python3
"""Config-driven HTCondor submission for the kspi_analysis pipeline (s1-s6).

Reads htcondor/htcondor.yaml, which holds every selector, per-stage CLI flag
and resource setting, and emits one HTCondor file set per stage. Each run of a
stage gets its own timestamped folder:

    htcondor/<stage>/run_<YYYYmmdd_HHMMSS>/<stage>.dag        node list (+ MAXJOBS)
    htcondor/<stage>/run_<YYYYmmdd_HHMMSS>/<node>.sub          one submit file per job
    htcondor/<stage>/run_<YYYYmmdd_HHMMSS>/logs/               job log/out/err

So each DAG run is self-contained and never collides with a previous run (no
rescue-DAG picked up by accident, no stale dagman files to delete).

Stages s1-s5 are per-sample: every job is pinned to exactly one RunID
(track, ycset, polarity, hlt1) so HTCondor parallelizes the run.  Stage 6
(combine) is a single job and is intentionally NOT in the chain.

Usage (from the repo root):

    python htcondor/generate_dag.py --all             # s1..s6 DAGs
    python htcondor/generate_dag.py --stage s2        # one stage only
    python htcondor/generate_dag.py --stage s2 --dry-run
    python htcondor/generate_dag.py --all --config /path/htcondor.yaml

    The DAG paths baked into each .dag/.sub are relative to the repo root
    (using this folder's name as a prefix), matching how the workers resolve
    REPO = one level up from this script.  Submit from the repo root.

Exit code is non-zero if --all is requested and any stage is disabled, so a
wrapper script can notice when the config has turned a stage off.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = HERE.name + "/"
STAGE_ORDER = ["s1", "s2", "s3", "s4", "s5", "s6"]

DEFAULT_OUT_ROOT = "/eos/user/a/ahulsber/scripts/analysis"


# --------------------------------------------------------------------------
# Minimal YAML-subset parser (no dependency).  Handles exactly the constructs
# used in htcondor.yaml: two-level nested maps, inline lists, scalars.
# If PyYAML is available it is preferred.
# --------------------------------------------------------------------------
def _parse_scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        pass
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _parse_line(line: str):
    line = line.split(" #", 1)[0].rstrip()
    if ":" not in line:
        return None
    key, _, val = line.partition(":")
    key = key.strip()
    val = val.strip()
    if val.startswith("["):
        if val.strip() == "[]":
            return key, []
        inner = val[1:val.rfind("]")]
        items = []
        for tok in inner.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if (tok.startswith('"') and tok.endswith('"')) or \
               (tok.startswith("'") and tok.endswith("'")):
                tok = tok[1:-1]
            items.append(tok)
        return key, items
    return key, _parse_scalar(val)


def _load_yaml(text: str) -> dict:
    cfg: dict = {}
    current: dict | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        indent = len(raw) - len(raw.lstrip())
        parsed = _parse_line(stripped)
        if parsed is None:
            continue
        key, val = parsed
        # ignore the section header line only if its value is a comment block
        if indent == 0:
            is_section = isinstance(val, dict)
            if not is_section and key in ("common",) + tuple(
                f"stage{i}" for i in range(1, 7)
            ) + ("env",) and val is None:
                cfg[key] = cfg.get(key, {})
                current = cfg[key]
                continue
            cfg[key] = val
            current = None
        else:
            if current is None:
                current = {}
                cfg.setdefault("_orphans", current)
            current[key] = val
    return cfg


def _load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    except Exception:
        return _load_yaml(path.read_text())


# --------------------------------------------------------------------------
# HLT1 aliases (raw key / short token / human label) -> raw key.
# --------------------------------------------------------------------------
HLT1_LABELS = {
    "": "no HLT1 cut",
    "KS_Hlt1TwoTrackKsDecision_TOS": "KS_TwoTracks_TOS",
    "Pip_Hlt1TrackMVADecision_TOS": "Pip_TrackMVA_TOS",
    "DpTIS": "Dp*TIS",
    "DpTIS_PipMVATOS": "Dp*TIS & Pip TOS",
    "DpTIS_KSTwoTracks": "Dp*TIS & KS TOS",
}
HLT1_TOKENS = {
    "": "noHlt1",
    "KS_Hlt1TwoTrackKsDecision_TOS": "KS_TwoTracks_TOS",
    "Pip_Hlt1TrackMVADecision_TOS": "Pip_TrackMVA_TOS",
    "DpTIS": "DpTIS",
    "DpTIS_PipMVATOS": "DpTIS_PipMVATOS",
    "DpTIS_KSTwoTracks": "DpTIS_KSTwoTracks",
}
_HLT_ALIASES: dict[str, str] = {k: k for k in HLT1_LABELS}
_HLT_ALIASES.update({v: k for k, v in HLT1_TOKENS.items()})
_HLT_ALIASES.update({v: k for k, v in HLT1_LABELS.items()})

ALL_HLT1 = list(HLT1_LABELS)

# Which stage calls which worker.
STAGE_WORKER = {
    "s1": "run_s1_skim.sh",
    "s2": "run_s2_reweight.sh",
    "s3": "run_s3_asymmetries.sh",
    "s4": "run_s4_statistics.sh",
    "s5": "run_s5_weighting_performance.sh",
    "s6": "run_s6_combine.sh",
}


def _token(hlt1: str) -> str:
    return HLT1_TOKENS[hlt1]


def _resolve_hlt1(chosen: list) -> list[str]:
    if not chosen:
        return list(ALL_HLT1)
    out: list[str] = []
    for c in chosen:
        if c not in _HLT_ALIASES:
            sys.exit(f"error: unknown hlt1 selection {c!r}\n"
                     f"       valid: {', '.join(sorted(HLT1_TOKENS.values()))} "
                     f"or a raw key")
        out.append(_HLT_ALIASES[c])
    return out


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _load_inventory():
    """Return an object with iter_samples(...) / n_files(...), preferring the
    repo's config.datasets and falling back to this folder's sample_inventory."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))

    import importlib
    for modname, want in (("config.datasets", "datasets"),
                          ("sample_inventory", "sample_inventory")):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            if modname == "sample_inventory":
                raise ImportError(
                    "could not load a sample inventory (config.datasets or "
                    f"sample_inventory.py): {exc}"
                ) from exc
            continue
        if modname == "config.datasets":
            return mod
        class _Inv:
            pass
        _Inv.iter_samples = staticmethod(mod.iter_samples)
        _Inv.n_files = staticmethod(mod.n_files)
        return _Inv()
    raise RuntimeError("no sample inventory available")


def _runids(inv, track, ycset, polarity):
    track_set = set(track) if track else None
    ycset_set = set(ycset) if ycset else None
    pol_set = set(polarity) if polarity else None
    return sorted(
        (t, y, p)
        for t, y, p, _n in inv.iter_samples()
        if (track_set is None or t in track_set)
        and (ycset_set is None or y in ycset_set)
        and (pol_set is None or p in pol_set)
    )


def _value(cfg: dict, section: str, key: str, default=None):
    sec = cfg.get(section) or {}
    return sec.get(key, default)


def _sec(stage: str) -> str:
    """Map a short stage id ('s1') to its config section name ('stage1')."""
    return "stage" + stage[1:]


# --------------------------------------------------------------------------
def _run_folder(stage: str, log_subdir: str, dry_run: bool) -> tuple[Path, str, str]:
    """Return a fresh per-run path for a stage without creating dirs if
    dry_run is set.  Returns (stage_dir, stage_prefix, run_stamp) so the
    caller uses exactly the path that is (or will be) created."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}"
    stage_dir = HERE / stage / run_name
    if not dry_run:
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / log_subdir).mkdir(parents=True, exist_ok=True)
    stage_prefix = PREFIX + str(stage_dir.relative_to(HERE)) + "/"
    return stage_dir, stage_prefix, stamp


def _flags_for_stage(stage: str, cfg: dict, runid, hlt1: str, common: dict) -> list:
    """All CLI words passed to this job, in order (track/ycset/polarity/hlt1
    pins first, then the stage-specific extra args, then common toggles)."""
    words: list[str] = []
    if stage == "s6":
        extra = _as_list(_value(cfg, "stage6", "args"))
        words += ["--name", str(_value(cfg, "stage6", "name", "nominal"))]
        words += ["--tag", str(common.get("tag", "v1"))]
        words += [str(w) for w in extra]
        return words

    t, y, p = runid
    words += ["--track", t, "--ycset", y, "--polarity", p, "--hlt1", hlt1]
    if stage != "s1":
        words += ["--tag", str(common.get("tag", "v1"))]

    stage_args = _as_list(_value(cfg, _sec(stage), "args"))
    words += [str(w) for w in stage_args]

    # Global toggles, gated by which flags each stage's CLI actually accepts.
    #   --no-mt     : all per-sample stages (s1..s5), not s6
    #   --overwrite : s1, s2, s3, s4 (s5/s6 do not define it)
    if common.get("no_mt") and stage in ("s1", "s2", "s3", "s4", "s5"):
        words.append("--no-mt")
    if common.get("overwrite") and stage in ("s1", "s2", "s3", "s4"):
        words.append("--overwrite")
    return words


def _write_sub(sub: Path, executable: str, args: list, resources: dict,
               repo: str, comment: str, cpus: int,
               worker_prefix: str, stage_prefix: str,
               out_root: str, run_path: str) -> None:
    arg_str = '"' + " ".join(args) + '"'
    sub.write_text(
        f"""\
# {comment}
universe              = vanilla
getenv                = true
notification          = never
executable            = {worker_prefix}{executable}
arguments             = {arg_str}
initialdir            = {repo}
environment           = "REPO={repo} KSPI_OUT_ROOT={out_root} KSPI_RUN_DIR={run_path}"
log                   = {stage_prefix}logs/{sub.stem}.log
output                = {stage_prefix}logs/{sub.stem}.out
error                 = {stage_prefix}logs/{sub.stem}.err
should_transfer_files = NO
+JobFlavour            = "{resources.get('flavour', 'longlunch')}"
RequestCpus          = {int(cpus)}
queue
"""
    )


def _stage_names(args) -> list[str]:
    if args.all:
        return STAGE_ORDER
    for name in args.stage:
        if name not in STAGE_ORDER:
            sys.exit(f"error: unknown stage {name!r} (expected one of "
                     f"{', '.join(STAGE_ORDER)})")
    return args.stage


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="generate every stage")
    p.add_argument("--stage", nargs="*", default=[],
                   help="which stage(s) to generate, e.g. --stage s3 "
                        "(or --all for everything)")
    p.add_argument("--config", default=None,
                   help="path to the YAML config "
                        "(default: htcondor/htcondor.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the job list, write nothing")
    p.add_argument("--clean", action="store_true",
                   help="remove stale <stage>.dag.* DAGMan artifacts first")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.stage:
        sys.exit("specify --stage sN (or --all) to choose what to generate")

    config_path = Path(args.config) if args.config else HERE / "htcondor.yaml"
    if not config_path.exists():
        sys.exit(f"config not found: {config_path}")
    cfg = _load_config(config_path)
    common = cfg.get("common") or {}
    env_cfg = cfg.get("env") or {}

    repo = str(common.get("repo") or Path.cwd().resolve())
    log_subdir = (env_cfg.get("log_dir") or "logs")
    out_root = (os.environ.get("KSPI_OUT_ROOT")
                or common.get("out_root")
                or DEFAULT_OUT_ROOT)
    out_root = str(out_root)

    track = _as_list(common.get("track"))
    ycset = _as_list(common.get("ycset"))
    polarity = _as_list(common.get("polarity"))
    hlt1_sel = _resolve_hlt1(_as_list(common.get("hlt1")))

    inv = _load_inventory()

    chosen = _stage_names(args)

    # Ensure s6 command is buildable even though it is never auto-chained.
    all_enabled = True
    for stage in chosen:
        enabled = _value(cfg, _sec(stage), "enabled", True)
        if stage != "s6" and not enabled:
            print(f"note: {stage} is disabled in the config; skipping")
            all_enabled = False

    if args.clean:
        for stage in chosen:
            stage_dir = HERE / stage
            for p in stage_dir.glob(f"{stage}.dag.*"):
                if p.is_file():
                    p.unlink()
                    print(f"removed {HERE.name}/{stage}/{p.name}")
            for p in HERE.glob(f"{stage}.dag.*"):  # legacy top-level artifacts
                if p.is_file():
                    p.unlink()
                    print(f"removed {HERE.name}/{p.name}")

    maxjobs = int(common.get("maxjobs", 16))
    worker_prefix = HERE.name + "/"
    totals: dict[str, int] = {}
    for stage in chosen:
        enabled = _value(cfg, _sec(stage), "enabled", True)
        if stage != "s6" and not enabled:
            totals[stage] = 0
            continue
        resources = {
            "memory": _value(cfg, _sec(stage), "memory", 4000),
            "flavour": _value(cfg, _sec(stage), "flavour", "longlunch"),
        }
        cpus = int(_value(cfg, _sec(stage), "cpus", 1))
        worker = STAGE_WORKER[stage]

        # Fresh folder per run: <htcondor>/<stage>/run_<timestamp>/
        stage_dir, stage_prefix, run_stamp = _run_folder(
            stage, log_subdir, args.dry_run
        )
        run_path = stage_prefix  # repo-root relative

        node_lines = [f"MAXJOBS * {maxjobs}"]
        count = 0

        if stage == "s6":
            args_list = _flags_for_stage(stage, cfg, None, "", common)
            node = "s6_combine"
            sub = stage_dir / f"{node}.sub"
            comment = f"stage-6 combine: {' '.join(args_list)}"
            if not args.dry_run:
                _write_sub(sub, worker, args_list, resources, repo, comment,
                           cpus, worker_prefix, stage_prefix, out_root, run_path)
            node_lines.append(f"JOB {node} {stage_prefix}{sub.name}")
            count = 1
        else:
            runids_ = _runids(inv, track, ycset, polarity)
            per_runid = []
            for h in hlt1_sel:
                for (t, y, p) in runids_:
                    args_list = _flags_for_stage(stage, cfg, (t, y, p), h, common)
                    node = f"{stage}_{t}_{y}_{p}_{_token(h)}"
                    sub = stage_dir / f"{node}.sub"
                    comment = (f"stage {stage}: track={t} ycset={y} "
                               f"polarity={p} hlt1={h!r}")
                    per_runid.append((sub, args_list, comment))
            for sub, args_list, comment in per_runid:
                if not args.dry_run:
                    _write_sub(sub, worker, args_list, resources, repo,
                               comment, cpus, worker_prefix, stage_prefix,
                               out_root, run_path)
                node_lines.append(
                    f"JOB {sub.stem} {stage_prefix}{sub.name}"
                )
            count = len(per_runid)
            if args.dry_run:
                print(f"stage {stage}: {count} job(s)")
        totals[stage] = count
        dag_path = stage_dir / f"{stage}.dag"
        if not args.dry_run:
            dag_path.write_text("\n".join(node_lines) + "\n")
            print(f"wrote {count:3d} job(s) -> {HERE.name}/{stage}/run_{run_stamp}/{stage}.dag")

    return 0 if all_enabled else 2


if __name__ == "__main__":
    raise SystemExit(main())