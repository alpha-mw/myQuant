#!/usr/bin/env python3
"""
离线公司文档语义快照存储。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_investor.branch_contracts import CorporateDocumentSnapshot


class CorporateDocumentStore:
    """读取/写入离线 corporate document semantic snapshots。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _symbol_path(self, symbol: str) -> Path:
        normalized = symbol.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{normalized}.json"

    def get_semantic_snapshot(self, symbol: str, as_of: str) -> CorporateDocumentSnapshot:
        path = self._symbol_path(symbol)
        neutral = CorporateDocumentSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=False,
            source="offline_snapshot",
            publish_time=f"{as_of}T00:00:00" if as_of else "",
            effective_time=f"{as_of}T00:00:00" if as_of else "",
            ingest_time="",
            revision_id=f"corporate_document:offline_doc_store:{as_of}",
            is_estimated=True,
            data_quality={
                "status": "neutral_snapshot",
                "reason": "provider_missing",
                "provider_missing": True,
                "provider_name": "offline_doc_store",
            },
            provenance={
                "snapshot_type": "corporate_document",
                "provider_name": "offline_doc_store",
                "reason": "provider_missing",
                "provider_missing": True,
            },
        )
        if not path.exists():
            neutral.notes.append("document_snapshot_missing")
            return neutral

        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = CorporateDocumentSnapshot(
            symbol=symbol,
            as_of=str(payload.get("as_of", as_of)),
            available=bool(payload.get("available", True)),
            source=str(payload.get("source", "offline_snapshot")),
            publish_time=str(payload.get("publish_time", f"{payload.get('as_of', as_of)}T00:00:00")),
            effective_time=str(payload.get("effective_time", f"{payload.get('as_of', as_of)}T00:00:00")),
            ingest_time=str(payload.get("ingest_time", "")),
            revision_id=str(payload.get("revision_id", f"corporate_document:offline_doc_store:{payload.get('as_of', as_of)}")),
            is_estimated=bool(payload.get("is_estimated", False)),
            data_quality=dict(payload.get("data_quality", {"provider_missing": False})),
            provenance=dict(payload.get("provenance", {"provider_missing": False})),
            latest_document_type=str(payload.get("latest_document_type", "")),
            semantic_sentiment=float(payload.get("semantic_sentiment", 0.0)),
            execution_confidence=float(payload.get("execution_confidence", 0.0)),
            governance_red_flag=float(payload.get("governance_red_flag", 0.0)),
            key_phrases=[str(item) for item in payload.get("key_phrases", [])],
            key_risks=[str(item) for item in payload.get("key_risks", [])],
            notes=[str(item) for item in payload.get("notes", [])],
        )
        return snapshot

    def save_semantic_snapshot(self, snapshot: CorporateDocumentSnapshot | dict[str, Any]) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(snapshot, CorporateDocumentSnapshot):
            payload = {
                "symbol": snapshot.symbol,
                "as_of": snapshot.as_of,
                "available": snapshot.available,
                "source": snapshot.source,
                "publish_time": snapshot.publish_time,
                "effective_time": snapshot.effective_time,
                "ingest_time": snapshot.ingest_time,
                "revision_id": snapshot.revision_id,
                "is_estimated": snapshot.is_estimated,
                "data_quality": dict(snapshot.data_quality),
                "provenance": dict(snapshot.provenance),
                "latest_document_type": snapshot.latest_document_type,
                "semantic_sentiment": snapshot.semantic_sentiment,
                "execution_confidence": snapshot.execution_confidence,
                "governance_red_flag": snapshot.governance_red_flag,
                "key_phrases": list(snapshot.key_phrases),
                "key_risks": list(snapshot.key_risks),
                "notes": list(snapshot.notes),
            }
            symbol = snapshot.symbol
        else:
            payload = dict(snapshot)
            symbol = str(payload.get("symbol", "unknown"))

        path = self._symbol_path(symbol)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
