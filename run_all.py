#!/usr/bin/env python
"""Reproduce every table and figure of the paper.

    python run_all.py                 # full run (about 15-40 min on a laptop)
    python run_all.py --quick         # smoke test: last 4 seasons, few forecast realisations
    python run_all.py --steps main pareto figures

Outputs are written to results/tables (CSV + LaTeX) and results/figures (PDF + PNG).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from avhw import io
from avhw.pipeline import Runner

ALL = ["prices", "forecast", "microclimate", "crop", "water", "pv", "main", "pareto", "ablation",
       "sensitivity", "surrogate", "lambda_sweep", "climate", "transfer", "transfer_lambda", "tomato", "figures"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(io.ROOT / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--steps", nargs="*", default=ALL, choices=ALL)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = io.load_config(a.config)
    np.random.seed(cfg["seed"])
    run = Runner(cfg, quick=a.quick)
    steps = a.steps
    # steps the later ones depend on are always run
    needed = ["forecast", "microclimate", "crop", "water"]
    if any(s in steps for s in ["main", "pareto", "ablation", "sensitivity", "surrogate", "lambda_sweep", "climate", "transfer", "transfer_lambda", "tomato", "figures"]):
        steps = sorted(set(steps) | set(needed) | ({"main"} if any(s in steps for s in ["pareto", "ablation", "sensitivity", "figures"]) else set()),
                       key=ALL.index)
    for s in steps:
        if s == "figures":
            from avhw import plots
            plots.make_all(run)
            run.log("figures written")
            continue
        getattr(run, {"crop": "step_crop_validation", "pv": "step_pv_validation"}.get(s, f"step_{s}"))()
    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "steps": steps, "quick": a.quick, "runtime_min": round((time.time() - t0) / 60, 2)}
    (run.out / "logs" / "run_meta.json").parent.mkdir(exist_ok=True, parents=True)
    (run.out / "logs" / "run_meta.json").write_text(json.dumps(meta, indent=1))
    run.log(f"done in {meta['runtime_min']} min")
    run.write_log()


if __name__ == "__main__":
    main()
