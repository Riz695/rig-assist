"""Training data: `make pairs` (or `uv run python training/make_pairs.py`).

The logic lives in `rig_assist.training.make_pairs` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.training.make_pairs import main

if __name__ == "__main__":
    sys.exit(main())
