"""Exact daily Theme source versions; parsing grants no source or provider authority."""

from copy import deepcopy

from quant_investor.operations.daily_contract import validate_ref
from ._common import IntelligenceError

THEME_SOURCE_V2 = "cn-daily-theme-evidence-source.v2"
THEME_SOURCE_FIELDS = frozenset(
    {"dc_plan", "dc_capture", "dc_partitions", "tdx_plan", "tdx_capture", "tdx_partitions"}
)


def validate_theme_descriptor(value: dict) -> dict:
    if type(value) is not dict or set(value) != THEME_SOURCE_FIELDS:
        raise IntelligenceError("daily Theme descriptor fields differ")
    validate_ref(value["dc_plan"])
    validate_ref(value["dc_capture"])
    for name in ("dc_partitions", "tdx_partitions"):
        rows = value[name]
        if type(rows) is not list or (name == "dc_partitions" and not rows):
            raise IntelligenceError("daily Theme partition references differ")
        for reference in rows:
            validate_ref(reference)
        if len({(r["path"], r["sha256"]) for r in rows}) != len(rows):
            raise IntelligenceError("daily Theme partition references duplicate")
    if value["tdx_plan"] is None or value["tdx_capture"] is None:
        if (
            value["tdx_plan"] is not None
            or value["tdx_capture"] is not None
            or value["tdx_partitions"]
        ):
            raise IntelligenceError("daily Theme fallback references are incomplete")
    else:
        validate_ref(value["tdx_plan"])
        validate_ref(value["tdx_capture"])
        if not value["tdx_partitions"]:
            raise IntelligenceError("daily Theme fallback partitions are absent")
    return deepcopy(value)


def split_theme_source(value: dict | None) -> tuple[dict | None, dict | None, bool]:
    """Return pool/focus descriptors and explicit v2 status, without fallback lookup."""
    if value is None:
        return None, None, False
    if type(value) is not dict:
        raise IntelligenceError("daily Theme source is not an object")
    if "schema_version" not in value:
        return validate_theme_descriptor(value), None, False
    if (
        set(value) != {"schema_version", "pool", "pcb_ai_hardware"}
        or value["schema_version"] != THEME_SOURCE_V2
    ):
        raise IntelligenceError("daily Theme source schema is unknown")
    pool = validate_theme_descriptor(value["pool"])
    focus = value["pcb_ai_hardware"]
    return pool, None if focus is None else validate_theme_descriptor(focus), True


def descriptor_refs(value):
    """Collect physical refs after the owning native descriptor has validated."""
    result = {}

    def visit(item):
        if type(item) is dict:
            if set(item) == {"path", "sha256"}:
                ref = validate_ref(item)
                if ref["path"] in result and result[ref["path"]] != ref["sha256"]:
                    raise IntelligenceError("source descriptor references conflict")
                result[ref["path"]] = ref["sha256"]
            else:
                for nested in item.values():
                    visit(nested)
        elif type(item) is list:
            for nested in item:
                visit(nested)

    visit(value)
    return [{"path": path, "sha256": digest} for path, digest in sorted(result.items())]
