"""Fine-tune the embedder: `make train` (or `uv run --group train --group eval python
training/train_embedder.py`).

The logic lives in `rig_assist.training.train_embedder` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.training.train_embedder import main

if __name__ == "__main__":
    sys.exit(main())
