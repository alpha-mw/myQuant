from __future__ import annotations

import argparse
from pathlib import Path

from quant_platform.config import load_config
from quant_platform.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="HS300 Momentum + Volatility Strategy Pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run_pipeline(config)
    print(f"Report saved to: {outputs.report_path}")
    print(f"Package saved to: {outputs.zip_path}")
    print(f"Best momentum weight: {outputs.best_weight:.2f}")


if __name__ == "__main__":
    main()
