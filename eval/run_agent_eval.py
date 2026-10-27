"""/ask vs /diagnose comparison: `make eval-agents`.

The logic lives in `rig_assist.evaluation.run_agents` so it is tested with the package;
this file is the entry point PLAN.md names. `--help` lists the options.
"""

import sys

from rig_assist.evaluation.run_agents import main

if __name__ == "__main__":
    sys.exit(main())
