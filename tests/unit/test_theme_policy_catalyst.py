from __future__ import annotations

import json
from pathlib import Path

from quant_investor.themes.policy import PolicyCatalystScanner


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


def _event(
    *,
    event_id: str = "evt",
    issuer: str = "State Council",
    policy_level: str = "central",
    policy_type: str = "guidance",
    publish_date: str = "2026-06-01",
    theme_tags: list[str] | None = None,
    industry_tags: list[str] | None = None,
    symbol_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "title": f"{event_id} policy",
        "issuer": issuer,
        "publish_date": publish_date,
        "effective_date": publish_date,
        "policy_level": policy_level,
        "policy_type": policy_type,
        "theme_tags": ["Semiconductor"] if theme_tags is None else list(theme_tags),
        "industry_tags": [] if industry_tags is None else list(industry_tags),
        "symbol_tags": [] if symbol_tags is None else list(symbol_tags),
        "evidence_text": "Local fixture evidence for deterministic policy catalyst scoring.",
        "source_url": "local://fixture",
    }


def _score(path: Path, *, as_of: str = "2026-06-15"):
    scanner = PolicyCatalystScanner(event_path=path, lookback_days=30)
    events = scanner.load_events()
    return scanner.score_theme(
        theme_id="industry::semiconductor",
        theme_name="Semiconductor",
        member_symbols=["000001.SZ", "000002.SZ"],
        as_of=as_of,
        events=events,
    )


def test_policy_event_jsonl_loads_local_rows(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "policy.jsonl",
        [
            _event(event_id="evt-1"),
            _event(event_id="evt-2", issuer="Ministry of Industry", policy_level="ministry"),
        ],
    )

    scanner = PolicyCatalystScanner(event_path=path)
    events = scanner.load_events()

    assert scanner.status == "success"
    assert [event.event_id for event in events] == ["evt-1", "evt-2"]
    assert events[0].theme_tags == ["Semiconductor"]


def test_central_ministry_authority_scores_above_local_and_association(tmp_path: Path) -> None:
    central = _score(
        _write_jsonl(
            tmp_path / "central.jsonl",
            [_event(event_id="central", issuer="State Council", policy_level="central")],
        )
    )
    local = _score(
        _write_jsonl(
            tmp_path / "local.jsonl",
            [_event(event_id="local", issuer="Shanghai Municipal Commission", policy_level="local")],
        )
    )
    association = _score(
        _write_jsonl(
            tmp_path / "association.jsonl",
            [
                _event(
                    event_id="association",
                    issuer="Semiconductor Industry Association",
                    policy_level="association",
                )
            ],
        )
    )

    assert central.authority_score > local.authority_score > association.authority_score


def test_explicit_theme_and_symbol_tags_raise_beneficiary_clarity(tmp_path: Path) -> None:
    explicit = _score(
        _write_jsonl(
            tmp_path / "explicit.jsonl",
            [
                _event(
                    event_id="explicit",
                    theme_tags=["Semiconductor"],
                    symbol_tags=["000001.SZ"],
                )
            ],
        )
    )
    generic = _score(
        _write_jsonl(
            tmp_path / "generic.jsonl",
            [
                _event(
                    event_id="generic",
                    theme_tags=[],
                    industry_tags=["Semiconductor"],
                    symbol_tags=[],
                )
            ],
        )
    )

    assert explicit.beneficiary_clarity > generic.beneficiary_clarity
    assert explicit.specificity_score > generic.specificity_score


def test_old_policy_recency_decays(tmp_path: Path) -> None:
    fresh = _score(
        _write_jsonl(
            tmp_path / "fresh.jsonl",
            [_event(event_id="fresh", publish_date="2026-06-10")],
        ),
        as_of="2026-06-15",
    )
    stale = _score(
        _write_jsonl(
            tmp_path / "stale.jsonl",
            [_event(event_id="stale", publish_date="2026-04-01")],
        ),
        as_of="2026-06-15",
    )

    assert fresh.recency_score > stale.recency_score
    assert stale.policy_stage == "stale"
    assert "policy_stale" in stale.risk_flags


def test_funding_pilot_standard_procurement_raise_components(tmp_path: Path) -> None:
    funded = _score(
        _write_jsonl(
            tmp_path / "funded.jsonl",
            [
                _event(
                    event_id="funded",
                    policy_type="funding pilot standard procurement",
                    symbol_tags=["000001.SZ"],
                )
            ],
        )
    )
    guidance = _score(
        _write_jsonl(
            tmp_path / "guidance.jsonl",
            [_event(event_id="guidance", policy_type="guidance")],
        )
    )

    assert funded.implementation_score > guidance.implementation_score
    assert funded.funding_score > guidance.funding_score
    assert funded.policy_score > guidance.policy_score
