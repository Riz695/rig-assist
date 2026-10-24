"""Retrieval eval: `make eval` (or `uv run --group eval python eval/run_retrieval_eval.py`).

The logic lives in `rig_assist.evaluation.run_retrieval` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.evaluation.run_retrieval import main

if __name__ == "__main__":
    sys.exit(main())
