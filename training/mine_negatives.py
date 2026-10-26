"""Hard negatives: `make negatives` (or `uv run python training/mine_negatives.py`).

The logic lives in `rig_assist.training.make_negatives` so it is tested with the package;
this file is the entry point. `--help` lists the options.
"""

import sys

from rig_assist.training.make_negatives import main

if __name__ == "__main__":
    sys.exit(main())
