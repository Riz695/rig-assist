"""ONNX export: `make export MODEL=<folder>` (or `uv run --group train python
training/export_onnx.py <folder>`).

The logic lives in `rig_assist.training.export_onnx` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.training.export_onnx import main

if __name__ == "__main__":
    sys.exit(main())
