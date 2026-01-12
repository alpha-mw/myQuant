from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zipfile import ZipFile


def package_outputs(output_dir: Path) -> Path:
    date_tag = datetime.now().strftime("%Y%m%d")
    zip_name = f"沪深300双因子策略_最优版本_{date_tag}.zip"
    zip_path = output_dir / zip_name
    with ZipFile(zip_path, "w") as zipf:
        for path in output_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zipf.write(path, arcname=path.name)
    return zip_path
