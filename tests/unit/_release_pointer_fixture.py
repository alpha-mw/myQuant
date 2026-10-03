"""A fake frozen release on disk plus a pointer file that names it."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

COMMIT = "a" * 40


def write_release_fixture(
    tmp_path: Path, *, workspace: Path | None = None, **overrides: str
) -> tuple[Path, dict[str, str]]:
    """Create install/checkout/context under ``tmp_path`` and return the pointer path."""

    workspace = workspace or tmp_path / "workspace"
    install = tmp_path / "authority" / f"{COMMIT}-unified-runtime" / "installs" / f"{COMMIT}-1"
    (install / "bin").mkdir(parents=True)
    python = install / "bin" / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    (install / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    checkout = tmp_path / "checkouts" / f"{COMMIT}-unified-runtime"
    checkout.mkdir(parents=True)
    context = (
        workspace / "data/private/cn_daily_maintenance/factor_loop_contexts" / f"{COMMIT}.json"
    )
    context.parent.mkdir(parents=True, exist_ok=True)
    context.write_bytes(b'{"release_commit":"' + COMMIT.encode() + b'"}')
    values = {
        "RELEASE_COMMIT": COMMIT,
        "RELEASE_INSTALL_DIR": str(install),
        "RELEASE_CHECKOUT_DIR": str(checkout),
        "RELEASE_INSTALL_INPUT_SHA256": "b" * 64,
        "FACTOR_LOOP_CONTEXT": os.path.relpath(context, workspace),
        "FACTOR_LOOP_CONTEXT_SHA256": hashlib.sha256(context.read_bytes()).hexdigest(),
        **overrides,
    }
    pointer = workspace / "operations/releases/active.env"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        "# test pointer\n" + "".join(f"{key}={value}\n" for key, value in values.items())
    )
    return pointer, values
