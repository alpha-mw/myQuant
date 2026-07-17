"""Sealed local PIT fact-package contract for v16 Codex Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .candidate_pipeline import CandidateUnion, build_candidate_union

STAGE1_REQUEST_SCHEMA = "v16.codex-stage1.request.v1"
TARGET_DEFINITION = "CN_20D_NET_EXCESS_VS_CSI300_GT_0"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def _strict_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{path} contains an empty object key")
            if key in result:
                raise ValueError(f"{path} contains a duplicate normalised key: {key}")
            result[key] = _strict_json_value(child, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _strict_json_value(child, path=f"{path}[{index}]") for index, child in enumerate(value)
        )
    raise ValueError(f"{path} contains unsupported value type {type(value)!r}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


@dataclass(frozen=True)
class PITFactRow:
    """Compact frozen Q/F/M facts for one Eligibility-approved symbol."""

    symbol: str
    stratum: str
    eligibility_receipt_sha256: str
    formal_quant_score: float
    quant_facts: Mapping[str, Any]
    fundamental_facts: Mapping[str, Any]
    macro_facts: Mapping[str, Any]

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("PIT fact symbol must be non-empty")
        object.__setattr__(self, "symbol", symbol)
        stratum = str(self.stratum).strip()
        if not stratum:
            raise ValueError("PIT fact stratum must be non-empty")
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(
            self,
            "eligibility_receipt_sha256",
            _require_sha256(
                self.eligibility_receipt_sha256,
                field_name="eligibility_receipt_sha256",
            ),
        )
        if not isfinite(self.formal_quant_score) or not -1.0 <= self.formal_quant_score <= 1.0:
            raise ValueError("formal_quant_score must be finite and within [-1, 1]")
        for field_name in ("quant_facts", "fundamental_facts", "macro_facts"):
            value = _strict_json_value(getattr(self, field_name), path=field_name)
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "stratum": self.stratum,
            "eligibility_receipt_sha256": self.eligibility_receipt_sha256,
            "formal_quant_score": self.formal_quant_score,
            "quant_facts": _plain(self.quant_facts),
            "fundamental_facts": _plain(self.fundamental_facts),
            "macro_facts": _plain(self.macro_facts),
        }


@dataclass(frozen=True)
class Stage1FactPackage:
    schema_version: str
    target_definition: str
    market: str
    cutoff_at: str
    expires_at: str
    pit_pointer_sha256: str
    rows: tuple[PITFactRow, ...]
    funnel_symbols: tuple[str, ...]
    universe_symbol_set_sha256: str
    funnel_symbol_set_sha256: str
    stratum_counts: Mapping[str, int]
    payload_sha256: str

    def to_dict(self, *, include_payload_sha: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target_definition": self.target_definition,
            "market": self.market,
            "cutoff_at": self.cutoff_at,
            "expires_at": self.expires_at,
            "pit_pointer_sha256": self.pit_pointer_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "funnel_symbols": list(self.funnel_symbols),
            "universe_symbol_set_sha256": self.universe_symbol_set_sha256,
            "funnel_symbol_set_sha256": self.funnel_symbol_set_sha256,
            "stratum_counts": dict(self.stratum_counts),
        }
        if include_payload_sha:
            payload["payload_sha256"] = self.payload_sha256
        return payload

    def verify(self) -> None:
        expected = _sha(self.to_dict(include_payload_sha=False))
        if expected != self.payload_sha256:
            raise ValueError("Stage 1 fact package payload SHA mismatch")


def _parse_aware_timestamp(value: str, *, field_name: str) -> datetime:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_stage1_fact_package(
    *,
    rows: Iterable[PITFactRow],
    funnel_symbols: Iterable[str],
    cutoff_at: str,
    expires_at: str,
    pit_pointer_sha256: str,
) -> Stage1FactPackage:
    """Seal the complete eligible universe after formal Quant and Funnel.

    The builder rejects duplicate/missing symbols and never truncates the
    Funnel.  Supplemental candidates are deliberately absent at this stage;
    they may only arrive in the validated Stage 1 response.
    """

    cutoff = _parse_aware_timestamp(cutoff_at, field_name="cutoff_at")
    expiry = _parse_aware_timestamp(expires_at, field_name="expires_at")
    if expiry <= cutoff:
        raise ValueError("expires_at must be later than cutoff_at")
    pit_sha = _require_sha256(pit_pointer_sha256, field_name="pit_pointer_sha256")

    materialised = tuple(rows)
    symbols = [row.symbol for row in materialised]
    if not materialised:
        raise ValueError("Stage 1 fact package requires at least one eligible symbol")
    if len(symbols) != len(set(symbols)):
        raise ValueError("Stage 1 fact package contains duplicate symbols")

    # The full universe is sorted by symbol so serialization is deterministic;
    # stratum metadata lets Codex process it in auditable layers.
    ordered_rows = tuple(sorted(materialised, key=lambda row: row.symbol))
    universe_symbols = tuple(row.symbol for row in ordered_rows)
    funnel: CandidateUnion = build_candidate_union(funnel_symbols, [])
    unknown_funnel = sorted(set(funnel.funnel_symbols) - set(universe_symbols))
    if unknown_funnel:
        raise ValueError(f"Funnel contains symbols outside the eligible universe: {unknown_funnel}")

    stratum_counts: dict[str, int] = {}
    for row in ordered_rows:
        stratum_counts[row.stratum] = stratum_counts.get(row.stratum, 0) + 1

    base = {
        "schema_version": STAGE1_REQUEST_SCHEMA,
        "target_definition": TARGET_DEFINITION,
        "market": "CN",
        "cutoff_at": cutoff.isoformat().replace("+00:00", "Z"),
        "expires_at": expiry.isoformat().replace("+00:00", "Z"),
        "pit_pointer_sha256": pit_sha,
        "rows": [row.to_dict() for row in ordered_rows],
        "funnel_symbols": list(funnel.funnel_symbols),
        "universe_symbol_set_sha256": _sha(sorted(universe_symbols)),
        "funnel_symbol_set_sha256": _sha(sorted(funnel.funnel_symbols)),
        "stratum_counts": dict(sorted(stratum_counts.items())),
    }
    package = Stage1FactPackage(
        schema_version=STAGE1_REQUEST_SCHEMA,
        target_definition=TARGET_DEFINITION,
        market="CN",
        cutoff_at=base["cutoff_at"],
        expires_at=base["expires_at"],
        pit_pointer_sha256=pit_sha,
        rows=ordered_rows,
        funnel_symbols=funnel.funnel_symbols,
        universe_symbol_set_sha256=base["universe_symbol_set_sha256"],
        funnel_symbol_set_sha256=base["funnel_symbol_set_sha256"],
        stratum_counts=MappingProxyType(dict(sorted(stratum_counts.items()))),
        payload_sha256=_sha(base),
    )
    package.verify()
    return package
