#!/usr/bin/env python3
"""
Entry point from project root:
  python scripts/run_semantic_search.py --glove /path/to/glove.6B.300d.txt --query "food delivery problem"

Delegates to ``src.advanced_model``. Defaults assume CWD is the project root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advanced_model import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
