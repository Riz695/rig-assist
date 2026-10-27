"""Write the API's OpenAPI schema to a JSON file, without starting a server.

The UI generates its TypeScript types from this file (`make ui-types`), and a test checks
the committed copy matches the code, so the UI and the API can't drift apart silently.

    uv run python -m rig_assist.api.openapi_export ui/openapi.json
"""

import json
import sys
from pathlib import Path
from typing import Any

from rig_assist.api.main import create_app
from rig_assist.config import Settings


def openapi_schema() -> dict[str, Any]:
    # Defaults only (no .env): the schema must not depend on whose machine made it.
    # Nothing connects at creation: Qdrant, models and the LLM are only used per request.
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    return app.openapi()


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m rig_assist.api.openapi_export OUTPUT.json", file=sys.stderr)
        return 2
    Path(argv[0]).write_text(render(openapi_schema()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
