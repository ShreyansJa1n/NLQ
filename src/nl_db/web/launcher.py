"""Console entry-point that boots `streamlit run` against our app file.

Usage:
    nl-db-ui                 # launches the Streamlit server on the default port
    nl-db-ui --port 9000     # any flag is forwarded to streamlit run
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli as st_cli  # imported lazily — slow import

    app_path = Path(__file__).resolve().parent / "app.py"
    forwarded = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app_path), *forwarded]
    return int(st_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
