"""Answer eval: `make eval-answers` (or `uv run --group eval python eval/run_answer_eval.py`).

The logic lives in `rig_assist.evaluation.run_answers` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.evaluation.run_answers import main

if __name__ == "__main__":
    sys.exit(main())
