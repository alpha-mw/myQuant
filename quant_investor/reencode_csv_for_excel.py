#!/usr/bin/env python3
"""
把 CSV 重写为 Excel 友好的 UTF-8-SIG 编码。

适用于已有 UTF-8 CSV 在 Excel 中打开时中文乱码的情况。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def iter_csv_files(target: Path) -> list[Path]:
    """收集目标路径下的 CSV 文件。"""
    if target.is_file():
        return [target] if target.suffix.lower() == ".csv" else []
    return sorted(path for path in target.rglob("*.csv") if path.is_file())


def reencode_csv(path: Path) -> bool:
    """
    将 CSV 重写为 UTF-8-SIG。

    Returns:
        是否实际处理了文件。
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    path.write_text(text, encoding="utf-8-sig", newline="")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="把 CSV 重写为 UTF-8-SIG 编码")
    parser.add_argument("paths", nargs="+", help="CSV 文件或目录路径")
    args = parser.parse_args()

    processed = 0
    for raw_path in args.paths:
        target = Path(raw_path).expanduser().resolve()
        if not target.exists():
            print(f"跳过不存在路径: {target}")
            continue

        csv_files = iter_csv_files(target)
        if not csv_files:
            print(f"未找到 CSV: {target}")
            continue

        for csv_path in csv_files:
            reencode_csv(csv_path)
            processed += 1
            print(f"已重写: {csv_path}")

    print(f"完成，共处理 {processed} 个 CSV 文件")


if __name__ == "__main__":
    main()
