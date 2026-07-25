"""Version-neutral private storage compatibility package.

Factor Governance v4 binds the exact ``quant_investor.codex_review.storage``
module path.  The retired review workflow is intentionally not imported here;
keeping package import storage-only prevents that neutral Factor dependency
from loading any retired decision protocol.
"""

from .storage import (
    CONTROL_MAX_BYTES,
    MAX_JSON_DEPTH,
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    REQUEST_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    DifferentBytesError,
    ProtocolError,
    StateConflictError,
    StrictJSONError,
    assert_cas,
    atomic_write_bytes,
    canonical_json_bytes,
    ensure_private_dir,
    ensure_private_root,
    parse_strict_json_bytes,
    read_private_bytes,
    read_strict_json,
    run_lock,
    sha256_bytes,
    sha256_file,
    write_exact_once,
)

__all__ = [
    "CONTROL_MAX_BYTES",
    "MAX_JSON_DEPTH",
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "REQUEST_MAX_BYTES",
    "RESPONSE_MAX_BYTES",
    "DifferentBytesError",
    "ProtocolError",
    "StateConflictError",
    "StrictJSONError",
    "assert_cas",
    "atomic_write_bytes",
    "canonical_json_bytes",
    "ensure_private_dir",
    "ensure_private_root",
    "parse_strict_json_bytes",
    "read_private_bytes",
    "read_strict_json",
    "run_lock",
    "sha256_bytes",
    "sha256_file",
    "write_exact_once",
]
