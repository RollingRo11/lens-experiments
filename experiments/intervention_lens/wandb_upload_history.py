"""Upload finished swap-training histories (``<lens>_fit_history.json`` from ``fit-swap``) to
W&B as runs, one per file, so runs trained before W&B logging existed sit beside the new ones.
Each row of the history is logged at its ``step``; the run's config records the file and that
it was logged after training. Runs in the jpp_lens venv with the ``tracking`` extra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", default="lens-training")
parser.add_argument("--history", nargs="+", required=True)
args = parser.parse_args()
for path in args.history:
    name = Path(path).name.removesuffix("_fit_history.json")
    history = json.loads(Path(path).read_text())
    run = wandb.init(project=args.project, name=name, config={"history_file": path, "logged_after_training": True})
    for entry in history:
        run.log(entry, step=int(entry["step"]))
    run.finish()
    print(f"{name}: {len(history)} rows", flush=True)
