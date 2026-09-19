#!/usr/bin/env python3
"""Compatibility file entrypoint; the producer lives in the installed package."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.market.cn_benchmark_capture import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
