"""Fixed contextual-validation profiles and installed Python component identity."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, seal_artifact, validate_artifact

from .errors import SystemContractError, SystemSecurityError
from .release import installed_code_manifest_sha256

BOOTSTRAP_VALIDATION_PROFILE: Final = "factor-bootstrap-contextual-validation"
PROSPECTIVE_VALIDATION_PROFILE: Final = "factor-prospective-contextual-validation"
STRICT_SOURCE_DECODER_ID: Final = "factor-strict-parquet-source-decoder"
CONTEXTUAL_VALIDATOR_PACKAGE: Final = "quant_investor.factors.governance"

MAXIMUM_BOOTSTRAP_FACTOR_SOURCE_BYTES: Final = 2 * 1024**3
MAXIMUM_PROSPECTIVE_FACTOR_SOURCE_BYTES: Final = 16 * 1024**3
MAXIMUM_TOTAL_FACTOR_SOURCE_BYTES: Final = MAXIMUM_PROSPECTIVE_FACTOR_SOURCE_BYTES
MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES: Final = 512 * 1024**2
MAXIMUM_DECODED_SOURCE_ROWS: Final = 10_000_000
MAXIMUM_DECODED_SOURCE_CELLS: Final = 100_000_000
MAXIMUM_VALIDATION_RSS_BYTES: Final = 512 * 1024**2
MAXIMUM_DECODE_RESERVATION_BYTES: Final = 256 * 1024**2
DECODE_MEMORY_BUDGET_BYTES: Final = 384 * 1024**2

_PACKAGE_ROOT: Final = Path(__file__).resolve().parents[1]
_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAX_CODE_FILE_BYTES: Final = 64 * 1024 * 1024
_DECODER_MODULES: Final = (
    "quant_investor.factors.governance.bootstrap",
    "quant_investor.factors.governance.common",
    "quant_investor.factors.governance.errors",
    "quant_investor.factors.governance.implementations",
    "quant_investor.factors.governance.source",
)

VALIDATION_PROFILES: Final = MappingProxyType(
    {
        BOOTSTRAP_VALIDATION_PROFILE: MappingProxyType(
            {
                "validation_lane": "BOOTSTRAP",
                "callback_module": "quant_investor.factors.governance.contextual",
                "callback_qualified_name": "validate_bootstrap_contextual_run",
                "maximum_total_factor_source_bytes": MAXIMUM_BOOTSTRAP_FACTOR_SOURCE_BYTES,
            }
        ),
        PROSPECTIVE_VALIDATION_PROFILE: MappingProxyType(
            {
                "validation_lane": "PROSPECTIVE",
                "callback_module": "quant_investor.factors.governance.contextual",
                "callback_qualified_name": "validate_prospective_contextual_run",
                "maximum_total_factor_source_bytes": MAXIMUM_PROSPECTIVE_FACTOR_SOURCE_BYTES,
            }
        ),
    }
)


def component_registry() -> dict[str, Any]:
    """Return the immutable registry; artifact/request bytes cannot extend it."""

    return {
        "domain": "myquant-contextual-validation-component-registry",
        "profiles": [
            {
                "validation_profile_id": profile_id,
                **dict(VALIDATION_PROFILES[profile_id]),
            }
            for profile_id in sorted(VALIDATION_PROFILES)
        ],
        "source_decoder": {
            "decoder_id": STRICT_SOURCE_DECODER_ID,
            "module_name": "quant_investor.factors.governance.source",
            "qualified_name": "decode_source_role",
            "module_names": list(_DECODER_MODULES),
            "allowed_source_formats": ["PARQUET"],
            "fallback_allowed": False,
        },
    }


COMPONENT_REGISTRY_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(component_registry())
).hexdigest()


def validation_profile(validation_profile_id: str) -> dict[str, Any]:
    try:
        profile = VALIDATION_PROFILES[validation_profile_id]
    except (KeyError, TypeError) as exc:
        raise SystemContractError("validation profile is not compiled") from exc
    return dict(profile)


def _stat_key(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_uid,
    )


def _verify_code_directory(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise SystemSecurityError("installed component directory is unsafe")


def _read_code(path: Path) -> tuple[bytes, dict[str, Any]]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _READ_FLAGS)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_CODE_FILE_BYTES
        ):
            raise SystemSecurityError("installed component source file is unsafe")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        if (
            _stat_key(before) != _stat_key(after)
            or _stat_key(after) != _stat_key(path_after)
            or len(raw) != after.st_size
        ):
            raise SystemSecurityError("installed component source changed during read")
        return raw, {
            "path": "quant_investor/" + path.relative_to(_PACKAGE_ROOT).as_posix(),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }
    except SystemSecurityError:
        raise
    except (OSError, ValueError) as exc:
        raise SystemSecurityError("installed component source cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _module_path(module_name: str) -> Path:
    prefix = "quant_investor."
    if type(module_name) is not str or not module_name.startswith(prefix):
        raise SystemContractError("installed component module is outside quant_investor")
    relative = module_name[len(prefix) :].split(".")  # noqa: E203
    if not relative or any(not part.isidentifier() for part in relative):
        raise SystemContractError("installed component module name is invalid")
    module_path = _PACKAGE_ROOT.joinpath(*relative).with_suffix(".py")
    package_path = _PACKAGE_ROOT.joinpath(*relative, "__init__.py")
    if module_path.is_file() and not module_path.is_symlink():
        return module_path
    if package_path.is_file() and not package_path.is_symlink():
        return package_path
    raise SystemSecurityError("installed component module is not a regular Python module")


def _package_modules(package_name: str) -> tuple[str, ...]:
    package_init = _module_path(package_name)
    package_root = package_init.parent
    _verify_code_directory(package_root)
    modules: list[str] = []
    for directory, directory_names, file_names in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        _verify_code_directory(current)
        retained: list[str] = []
        for name in sorted(directory_names):
            if name == "__pycache__":
                continue
            child = current / name
            if child.is_symlink():
                raise SystemSecurityError("installed component package contains a symlink")
            _verify_code_directory(child)
            retained.append(name)
        directory_names[:] = retained
        for filename in sorted(file_names):
            if not filename.endswith(".py"):
                continue
            path = current / filename
            if path.is_symlink():
                raise SystemSecurityError("installed component package contains a symlink")
            relative = path.relative_to(package_root)
            suffix_parts = list(relative.parts)
            if suffix_parts[-1] == "__init__.py":
                suffix_parts = suffix_parts[:-1]
            else:
                suffix_parts[-1] = suffix_parts[-1][:-3]
            module = ".".join([package_name, *suffix_parts])
            modules.append(module)
    if not modules:
        raise SystemSecurityError("installed component package is empty")
    return tuple(sorted(set(modules)))


def ast_entrypoint_sha256(module_name: str, qualified_name: str) -> str:
    """Hash an exact function AST plus its canonical module/qualified identity."""

    if type(qualified_name) is not str or not qualified_name:
        raise SystemContractError("installed component qualified name is invalid")
    raw, _ = _read_code(_module_path(module_name))
    try:
        tree = ast.parse(raw.decode("utf-8", errors="strict"))
    except (SyntaxError, UnicodeError) as exc:
        raise SystemSecurityError("installed component Python source is invalid") from exc
    nodes: Sequence[ast.stmt] = tree.body
    selected: ast.AST | None = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if selected is None:
            raise SystemContractError("installed component entrypoint AST is absent")
        nodes = selected.body if hasattr(selected, "body") else ()
    if not isinstance(selected, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise SystemContractError("installed component entrypoint is not a function")
    preimage = {
        "domain": "myquant-python-ast-entrypoint",
        "module_name": module_name,
        "qualified_name": qualified_name,
        "node": ast.dump(selected, annotate_fields=True, include_attributes=False),
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _component_projection(
    *,
    component_id: str,
    component_registry_sha256: str,
    component_role: str,
    package_name: str,
    module_names: Sequence[str],
    entrypoints: Sequence[Mapping[str, str]],
    files: Sequence[Mapping[str, Any]],
    release_manifest_ref: Mapping[str, Any],
    installed_code_manifest_sha256: str,
    allowed_source_formats: Sequence[str],
    fallback_allowed: bool,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_registry_sha256": component_registry_sha256,
        "component_role": component_role,
        "package_name": package_name,
        "module_names": list(module_names),
        "entrypoints": [dict(row) for row in entrypoints],
        "files": [dict(row) for row in files],
        "release_manifest_ref": dict(release_manifest_ref),
        "installed_code_manifest_sha256": installed_code_manifest_sha256,
        "allowed_source_formats": list(allowed_source_formats),
        "fallback_allowed": fallback_allowed,
    }


def derive_installed_component_payload(
    *,
    component_id: str,
    component_role: str,
    package_name: str,
    module_names: Sequence[str],
    entrypoint_specs: Sequence[tuple[str, str]],
    release_manifest_ref: Mapping[str, Any],
    allowed_source_formats: Sequence[str] = (),
    fallback_allowed: bool = False,
) -> dict[str, Any]:
    """Securely derive an exact component payload from installed regular Python."""

    modules = tuple(sorted(set(module_names)))
    if not modules or tuple(module_names) != modules:
        raise SystemContractError("installed component modules must be sorted and unique")
    if package_name == CONTEXTUAL_VALIDATOR_PACKAGE and component_role == "CONTEXTUAL_VALIDATOR":
        if modules != _package_modules(package_name):
            raise SystemContractError("contextual validator must bind the whole package")
    paths = tuple(_module_path(module_name) for module_name in modules)
    file_rows = [_read_code(path)[1] for path in paths]
    if len({row["path"] for row in file_rows}) != len(file_rows):
        raise SystemContractError("installed component file projection is duplicated")
    file_rows.sort(key=lambda row: row["path"])
    entrypoints = [
        {
            "module_name": module,
            "qualified_name": qualified,
            "code_sha256": ast_entrypoint_sha256(module, qualified),
        }
        for module, qualified in entrypoint_specs
    ]
    entrypoints.sort(key=lambda row: (row["module_name"], row["qualified_name"]))
    if len({(row["module_name"], row["qualified_name"]) for row in entrypoints}) != len(
        entrypoints
    ):
        raise SystemContractError("installed component entrypoints are duplicated")
    formats = list(allowed_source_formats)
    if formats != sorted(set(formats)):
        raise SystemContractError("installed component source formats are not canonical")
    if type(fallback_allowed) is not bool:
        raise SystemContractError("installed component fallback flag is invalid")
    code_manifest = installed_code_manifest_sha256()
    projection = _component_projection(
        component_id=component_id,
        component_registry_sha256=COMPONENT_REGISTRY_SHA256,
        component_role=component_role,
        package_name=package_name,
        module_names=modules,
        entrypoints=entrypoints,
        files=file_rows,
        release_manifest_ref=release_manifest_ref,
        installed_code_manifest_sha256=code_manifest,
        allowed_source_formats=formats,
        fallback_allowed=fallback_allowed,
    )
    component_sha = hashlib.sha256(
        canonical_json_bytes({"domain": "myquant-installed-component", **projection})
    ).hexdigest()
    manifest_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-installed-component-manifest-id",
                "component_id": component_id,
                "component_sha256": component_sha,
                "release_manifest_ref": release_manifest_ref,
            }
        )
    ).hexdigest()
    return {
        "component_manifest_id": manifest_id,
        **projection,
        "component_sha256": component_sha,
        "outcome": "VALIDATED",
        "authority": "NON_AUTHORIZING",
    }


def seal_installed_component_manifest(*, created_at: str, **values: Any) -> dict[str, Any]:
    return seal_artifact(
        "system.installed_component_manifest",
        derive_installed_component_payload(**values),
        created_at=created_at,
    )


def validate_installed_component_manifest(
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    document = validate_artifact(artifact, expected_kind="system.installed_component_manifest")
    payload = document["payload"]
    expected = derive_installed_component_payload(
        component_id=payload["component_id"],
        component_role=payload["component_role"],
        package_name=payload["package_name"],
        module_names=payload["module_names"],
        entrypoint_specs=[
            (row["module_name"], row["qualified_name"]) for row in payload["entrypoints"]
        ],
        release_manifest_ref=payload["release_manifest_ref"],
        allowed_source_formats=payload["allowed_source_formats"],
        fallback_allowed=payload["fallback_allowed"],
    )
    if payload != expected:
        raise SystemContractError("installed component identity has drifted")
    return document


__all__ = [
    "BOOTSTRAP_VALIDATION_PROFILE",
    "COMPONENT_REGISTRY_SHA256",
    "CONTEXTUAL_VALIDATOR_PACKAGE",
    "DECODE_MEMORY_BUDGET_BYTES",
    "MAXIMUM_BOOTSTRAP_FACTOR_SOURCE_BYTES",
    "MAXIMUM_DECODED_SOURCE_CELLS",
    "MAXIMUM_DECODED_SOURCE_ROWS",
    "MAXIMUM_DECODE_RESERVATION_BYTES",
    "MAXIMUM_FACTOR_SOURCE_OBJECT_BYTES",
    "MAXIMUM_PROSPECTIVE_FACTOR_SOURCE_BYTES",
    "MAXIMUM_TOTAL_FACTOR_SOURCE_BYTES",
    "MAXIMUM_VALIDATION_RSS_BYTES",
    "PROSPECTIVE_VALIDATION_PROFILE",
    "STRICT_SOURCE_DECODER_ID",
    "VALIDATION_PROFILES",
    "ast_entrypoint_sha256",
    "component_registry",
    "derive_installed_component_payload",
    "seal_installed_component_manifest",
    "validate_installed_component_manifest",
    "validation_profile",
]
