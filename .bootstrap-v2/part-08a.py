from pathlib import Path

Path("src/sgchords/cli.py").write_text(
    """from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .audio import is_youtube_url
from .chords import suggest_capo
from .exports import render_bar_chart, render_summary, write_exports
from .service import analyze_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
""",
    encoding="utf-8",
)
