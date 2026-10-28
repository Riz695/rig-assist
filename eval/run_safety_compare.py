"""Blocking precision of the safety node across cached /diagnose runs: `make eval-safety`.

The logic lives in `rig_assist.evaluation.run_safety_compare` so it is tested with the
package; this file is the entry point. `--help` lists the options.
"""

import sys

from rig_assist.evaluation.run_safety_compare import main

if __name__ == "__main__":
    sys.exit(main())
