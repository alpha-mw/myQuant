"""Typed V17 v4 Top24 and Deep evidence closure.

Deep is a research-only shrink/veto layer.  It cannot create a positive
target delta, and an unavailable Top24 row is always a zero BUY veto.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)

FUSION_TOP24_VERSION: Final = "myquant.v17.v4.fusion-top24.v1"
DEEP_BUNDLE_VERSION: Final = (
    "myquant.v17.v4.deep-evidence-bundle.v1"
)
OFFICIAL_EVIDENCE_VERSION: Final = (
    "myquant.v17.v4.official-evidence.v1"
)
ISSUER_DOSSIER_VERSION: Final = (
    "myquant.v17.v4.issuer-dossier.v1"
)
EVENT_SCAN_VERSION: Final = "myquant.v17.v4.event-scan.v1"
PROMOTION_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.fusion-promotion-receipt.v1"
)
TOP_N: Final = 24
MAX_DOSSIER_AGE_DAYS: Final = 30
MAX_EVENT_SCAN_AGE_DAYS: Final = 7
MAX_DEEP_PENALTY: Final = Decimal("0.10")
DEEP_PENALTY_SCALE: Final = Decimal("0.10")
_REF_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
)
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class DeepClosureError(ValueError):
    """Raised when Top24 or Deep evidence cannot close exactly."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise DeepClosureError(f"DEEP_CLOSURE_BLOCKED:{reason}")


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) not in {str, int, float, Decimal} or type(value) is bool:
        _blocked(f"{label}_invalid")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_invalid")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _blocked("decimal_nonfinite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _date(value: Any, *, label: str) -> date:
    if type(value) is not str:
        _blocked(f"{label}_invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked(f"{label}_invalid")
    if parsed.isoformat() != value:
        _blocked(f"{label}_noncanonical")
    return parsed


def _validate_ref(
    value: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff: str,
    expected_version: str,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_REF_FIELDS):
        _blocked(f"{label}_shape")
    try:
        identity = require_opaque_id(
            value["artifact_id"],
            label=f"{label}.artifact_id",
        )
        version = require_opaque_id(
            value["artifact_version"],
            label=f"{label}.artifact_version",
        )
        ref_cutoff = require_utc_timestamp(
            value["cutoff"],
            label=f"{label}.cutoff",
        )
        require_sha256(
            value["byte_sha256"],
            label=f"{label}.byte_sha256",
        )
        require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError:
        _blocked(f"{label}_identity")
    if (
        version != expected_version
        or value["strategy_id"] != strategy_id
        or ref_cutoff > cutoff
    ):
        _blocked(f"{label}_binding")
    path = value["relative_path"]
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _blocked(f"{label}_path")
    return {
        "artifact_id": identity,
        **{
            field: str(value[field])
            for field in _REF_FIELDS
            if field != "artifact_id"
        },
    }


def _identity(document: Mapping[str, Any]) -> Any:
    return next(
        (
            document.get(field)
            for field in (
                "bundle_id",
                "dossier_id",
                "evidence_id",
                "output_id",
                "receipt_id",
                "scan_id",
            )
            if field in document
        ),
        None,
    )


def _read_exact(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> dict[str, Any]:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise DeepClosureError(
            f"DEEP_CLOSURE_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    try:
        document = load_canonical_resource(raw, label=label)
        sealed = validate_semantic_sha(document)
        validate_artifact(
            sealed,
            artifact_loader=artifact_loader,
        )
    except (CanonicalContractError, ValueError, RuntimeError):
        _blocked(f"{label}_native_validation")
    if (
        sealed.get("version") != reference["artifact_version"]
        or sealed.get("strategy_id") != reference["strategy_id"]
        or sealed.get("cutoff") != reference["cutoff"]
        or sealed.get("semantic_sha256")
        != reference["semantic_sha256"]
        or _identity(sealed) != reference["artifact_id"]
    ):
        _blocked(f"{label}_document_binding")
    return sealed


def artifact_ref(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    if (
        type(artifact) is not dict
        or type(relative_path) is not str
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(
            part in {"", ".", ".."}
            for part in relative_path.split("/")
        )
    ):
        _blocked("artifact_ref_invalid")
    identity = _identity(artifact)
    if type(identity) is not str:
        _blocked("artifact_ref_identity")
    return {
        "artifact_id": identity,
        "artifact_version": str(artifact.get("version")),
        "byte_sha256": hashlib.sha256(
            canonical_bytes(artifact) + b"\n"
        ).hexdigest(),
        "cutoff": str(artifact.get("cutoff")),
        "relative_path": relative_path,
        "semantic_sha256": str(artifact.get("semantic_sha256")),
        "strategy_id": str(artifact.get("strategy_id")),
    }


@dataclass(frozen=True)
class FusionTop24Input:
    symbol: str
    fused_score: Any
    base_target: Any


@dataclass(frozen=True)
class DeepEvidenceInput:
    symbol: str
    status: str
    official_evidence_refs: tuple[Mapping[str, Any], ...] = ()
    issuer_dossier_ref: Mapping[str, Any] | None = None
    event_scan_ref: Mapping[str, Any] | None = None
    signal: Any | None = None
    buy_veto: bool = False
    reason: str = ""


def build_fusion_top24(
    rows: Sequence[FusionTop24Input],
    *,
    output_id: str,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    created_at: str,
    promotion_receipt_ref: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    try:
        output = require_opaque_id(output_id, label="output_id")
        run = require_opaque_id(run_id, label="run_id")
        strategy = require_opaque_id(
            strategy_id,
            label="strategy_id",
        )
        cutoff_text = require_utc_timestamp(cutoff, label="cutoff")
        created = require_utc_timestamp(
            created_at,
            label="created_at",
        )
    except IdentityContractError:
        _blocked("fusion_top24_identity")
    if created < cutoff_text or len(rows) != TOP_N:
        _blocked("fusion_top24_shape")
    promotion_ref = _validate_ref(
        promotion_receipt_ref,
        strategy_id=strategy,
        cutoff=cutoff_text,
        expected_version=PROMOTION_RECEIPT_VERSION,
        label="promotion_receipt_ref",
    )
    promotion = _read_exact(
        promotion_ref,
        artifact_loader=artifact_loader,
        label="promotion_receipt",
    )
    if (
        promotion["accepted"] is not True
        or promotion["status"] != "PROMOTED"
        or promotion["run_id"] != run
    ):
        _blocked("fusion_promotion_not_active")
    native_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        if (
            not isinstance(row, FusionTop24Input)
            or type(row.symbol) is not str
            or row.symbol in seen
        ):
            _blocked("fusion_top24_domain")
        seen.add(row.symbol)
        score = _decimal(
            row.fused_score,
            label=f"{row.symbol}.fused_score",
        )
        target = _decimal(
            row.base_target,
            label=f"{row.symbol}.base_target",
        )
        if target < 0:
            _blocked("fusion_top24_negative_target")
        native_rows.append(
            {
                "base_target": _decimal_text(target),
                "fused_score": _decimal_text(score),
                "rank": rank,
                "symbol": row.symbol,
            }
        )
    artifact = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "created_at": created,
            "cutoff": cutoff_text,
            "output_id": output,
            "promotion_receipt_ref": promotion_ref,
            "protocol_version": PROTOCOL_VERSION,
            "rows": native_rows,
            "run_id": run,
            "strategy_id": strategy,
            "version": FUSION_TOP24_VERSION,
        }
    )
    try:
        validate_artifact(artifact)
    except ValueError:
        _blocked("fusion_top24_native_validation")
    return artifact


def _complete_deep_row(
    item: DeepEvidenceInput,
    *,
    base_target: Decimal,
    strategy_id: str,
    cutoff: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    if (
        not item.official_evidence_refs
        or item.issuer_dossier_ref is None
        or item.event_scan_ref is None
        or item.signal is None
        or type(item.buy_veto) is not bool
        or item.reason
    ):
        _blocked(f"complete_evidence_shape:{item.symbol}")
    official_refs = tuple(
        _validate_ref(
            reference,
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version=OFFICIAL_EVIDENCE_VERSION,
            label=f"{item.symbol}.official_evidence_refs[{index}]",
        )
        for index, reference in enumerate(
            item.official_evidence_refs
        )
    )
    identities = [
        (
            reference["relative_path"],
            reference["byte_sha256"],
            reference["artifact_id"],
        )
        for reference in official_refs
    ]
    if identities != sorted(identities) or len(identities) != len(
        set(identities)
    ):
        _blocked(f"official_evidence_ref_order:{item.symbol}")
    official = [
        _read_exact(
            reference,
            artifact_loader=artifact_loader,
            label=f"{item.symbol}.official_evidence[{index}]",
        )
        for index, reference in enumerate(official_refs)
    ]
    if any(document["symbol"] != item.symbol for document in official):
        _blocked(f"official_evidence_symbol:{item.symbol}")
    dossier_ref = _validate_ref(
        item.issuer_dossier_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=ISSUER_DOSSIER_VERSION,
        label=f"{item.symbol}.issuer_dossier_ref",
    )
    event_ref = _validate_ref(
        item.event_scan_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=EVENT_SCAN_VERSION,
        label=f"{item.symbol}.event_scan_ref",
    )
    dossier = _read_exact(
        dossier_ref,
        artifact_loader=artifact_loader,
        label=f"{item.symbol}.issuer_dossier",
    )
    event = _read_exact(
        event_ref,
        artifact_loader=artifact_loader,
        label=f"{item.symbol}.event_scan",
    )
    cutoff_date = _date(cutoff[:10], label="cutoff_date")
    dossier_date = _date(dossier["as_of"], label="dossier.as_of")
    event_date = _date(event["as_of"], label="event_scan.as_of")
    if (
        dossier["symbol"] != item.symbol
        or event["symbol"] != item.symbol
        or not 0
        <= (cutoff_date - dossier_date).days
        <= MAX_DOSSIER_AGE_DAYS
        or not 0
        <= (cutoff_date - event_date).days
        <= MAX_EVENT_SCAN_AGE_DAYS
        or not set(
            reference["byte_sha256"]
            for reference in dossier["official_evidence_refs"]
        ).issubset(
            reference["byte_sha256"] for reference in official_refs
        )
        or not set(
            reference["byte_sha256"]
            for reference in event["official_evidence_refs"]
        ).issubset(
            reference["byte_sha256"] for reference in official_refs
        )
    ):
        _blocked(f"deep_freshness_or_lineage:{item.symbol}")
    signal = _decimal(item.signal, label=f"{item.symbol}.signal")
    with localcontext() as context:
        context.prec = 40
        penalty = min(
            MAX_DEEP_PENALTY,
            max(
                Decimal("0"),
                DEEP_PENALTY_SCALE
                * max(-signal, Decimal("0")),
            ),
        )
        target = (
            Decimal("0")
            if item.buy_veto
            else base_target * (Decimal("1") - penalty)
        )
    if target > base_target:
        _blocked(f"deep_positive_delta:{item.symbol}")
    return {
        "buy_veto": item.buy_veto,
        "event_scan_ref": event_ref,
        "issuer_dossier_ref": dossier_ref,
        "official_evidence_refs": list(official_refs),
        "reason": "",
        "signal": _decimal_text(signal),
        "status": "COMPLETE",
        "symbol": item.symbol,
        "target_after_deep": _decimal_text(target),
    }


def build_deep_evidence_bundle(
    inputs: Sequence[DeepEvidenceInput],
    *,
    bundle_id: str,
    fusion_top24_ref: Mapping[str, Any],
    created_at: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    try:
        bundle = require_opaque_id(bundle_id, label="bundle_id")
    except IdentityContractError:
        _blocked("deep_bundle_identity")
    if len(inputs) != TOP_N:
        _blocked("deep_top24_row_count")
    raw_ref = dict(fusion_top24_ref)
    strategy = raw_ref.get("strategy_id")
    cutoff = raw_ref.get("cutoff")
    if type(strategy) is not str or type(cutoff) is not str:
        _blocked("fusion_top24_ref_identity")
    fusion_ref = _validate_ref(
        raw_ref,
        strategy_id=strategy,
        cutoff=cutoff,
        expected_version=FUSION_TOP24_VERSION,
        label="fusion_top24_ref",
    )
    fusion = _read_exact(
        fusion_ref,
        artifact_loader=artifact_loader,
        label="fusion_top24",
    )
    try:
        created = require_utc_timestamp(
            created_at,
            label="created_at",
        )
    except IdentityContractError:
        _blocked("deep_bundle_created_at")
    if created < cutoff:
        _blocked("deep_bundle_created_at_before_cutoff")
    top_rows = fusion["rows"]
    top_symbols = [row["symbol"] for row in top_rows]
    input_symbols = [item.symbol for item in inputs]
    if input_symbols != top_symbols or len(set(input_symbols)) != TOP_N:
        _blocked("deep_top24_exact_domain")
    rows: list[dict[str, Any]] = []
    for item, top_row in zip(inputs, top_rows, strict=True):
        if not isinstance(item, DeepEvidenceInput):
            _blocked("deep_input_type")
        base_target = _decimal(
            top_row["base_target"],
            label=f"{item.symbol}.base_target",
        )
        if item.status == "UNAVAILABLE":
            if (
                item.official_evidence_refs
                or item.issuer_dossier_ref is not None
                or item.event_scan_ref is not None
                or item.signal is not None
                or item.buy_veto is not True
                or type(item.reason) is not str
                or not item.reason
            ):
                _blocked(f"unavailable_evidence_shape:{item.symbol}")
            rows.append(
                {
                    "buy_veto": True,
                    "event_scan_ref": None,
                    "issuer_dossier_ref": None,
                    "official_evidence_refs": [],
                    "reason": item.reason,
                    "signal": None,
                    "status": "UNAVAILABLE",
                    "symbol": item.symbol,
                    "target_after_deep": "0",
                }
            )
        elif item.status == "COMPLETE":
            rows.append(
                _complete_deep_row(
                    item,
                    base_target=base_target,
                    strategy_id=strategy,
                    cutoff=cutoff,
                    artifact_loader=artifact_loader,
                )
            )
        else:
            _blocked(f"deep_status:{item.symbol}")
    artifact = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "bundle_id": bundle,
            "created_at": created,
            "cutoff": cutoff,
            "fusion_top24_ref": fusion_ref,
            "protocol_version": PROTOCOL_VERSION,
            "rows": rows,
            "run_id": fusion["run_id"],
            "state": "DEEP_COMPLETE",
            "strategy_id": strategy,
            "version": DEEP_BUNDLE_VERSION,
        }
    )
    try:
        validate_artifact(artifact)
    except ValueError:
        _blocked("deep_bundle_native_validation")
    return artifact


def revalidate_deep_evidence_bundle(
    reference: Mapping[str, Any],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen a Deep bundle and every native evidence byte it transitively binds."""

    raw_ref = dict(reference)
    strategy = raw_ref.get("strategy_id")
    cutoff = raw_ref.get("cutoff")
    if type(strategy) is not str or type(cutoff) is not str:
        _blocked("deep_bundle_ref_identity")
    bundle_ref = _validate_ref(
        raw_ref,
        strategy_id=strategy,
        cutoff=cutoff,
        expected_version=DEEP_BUNDLE_VERSION,
        label="deep_bundle_ref",
    )
    bundle = _read_exact(
        bundle_ref,
        artifact_loader=artifact_loader,
        label="deep_bundle",
    )
    fusion_ref = _validate_ref(
        bundle["fusion_top24_ref"],
        strategy_id=strategy,
        cutoff=cutoff,
        expected_version=FUSION_TOP24_VERSION,
        label="deep_bundle.fusion_top24_ref",
    )
    fusion = _read_exact(
        fusion_ref,
        artifact_loader=artifact_loader,
        label="deep_bundle.fusion_top24",
    )
    if bundle["run_id"] != fusion["run_id"]:
        _blocked("deep_bundle_run_binding")
    rows = bundle["rows"]
    top_rows = fusion["rows"]
    if (
        len(rows) != TOP_N
        or len(top_rows) != TOP_N
        or [row["symbol"] for row in rows]
        != [row["symbol"] for row in top_rows]
    ):
        _blocked("deep_bundle_top24_domain")
    for row, top_row in zip(rows, top_rows, strict=True):
        if row["status"] == "UNAVAILABLE":
            expected = {
                "buy_veto": True,
                "event_scan_ref": None,
                "issuer_dossier_ref": None,
                "official_evidence_refs": [],
                "reason": row["reason"],
                "signal": None,
                "status": "UNAVAILABLE",
                "symbol": row["symbol"],
                "target_after_deep": "0",
            }
        else:
            expected = _complete_deep_row(
                DeepEvidenceInput(
                    symbol=row["symbol"],
                    status="COMPLETE",
                    official_evidence_refs=tuple(
                        row["official_evidence_refs"]
                    ),
                    issuer_dossier_ref=row["issuer_dossier_ref"],
                    event_scan_ref=row["event_scan_ref"],
                    signal=row["signal"],
                    buy_veto=row["buy_veto"],
                    reason=row["reason"],
                ),
                base_target=_decimal(
                    top_row["base_target"],
                    label=f"{row['symbol']}.base_target",
                ),
                strategy_id=strategy,
                cutoff=cutoff,
                artifact_loader=artifact_loader,
            )
        if row != expected:
            _blocked(f"deep_bundle_row_replay:{row['symbol']}")
    return bundle, fusion


__all__ = [
    "DEEP_BUNDLE_VERSION",
    "EVENT_SCAN_VERSION",
    "FUSION_TOP24_VERSION",
    "ISSUER_DOSSIER_VERSION",
    "OFFICIAL_EVIDENCE_VERSION",
    "DeepClosureError",
    "DeepEvidenceInput",
    "FusionTop24Input",
    "artifact_ref",
    "build_deep_evidence_bundle",
    "build_fusion_top24",
    "revalidate_deep_evidence_bundle",
]
