"""Helper to launch the Streamlit application when packaged."""

import os
import sys

import streamlit.web.cli as stcli


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.getcwd(), path))


def main() -> int:
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
