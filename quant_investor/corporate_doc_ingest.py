#!/usr/bin/env python3
"""
离线公司文档语义快照写入器。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_investor.branch_contracts import CorporateDocumentSnapshot
from quant_investor.corporate_doc_store import CorporateDocumentStore


class CorporateDocumentIngestor:
    """将外部预处理后的文档语义结果写入离线 snapshot store。"""

    def __init__(self, base_dir: str | Path):
        self._store = CorporateDocumentStore(base_dir=base_dir)

    def ingest_snapshot(
        self,
        symbol: str,
        as_of: str,
        payload: dict[str, Any],
    ) -> Path:
        snapshot = CorporateDocumentSnapshot(
            symbol=symbol,
            as_of=as_of,
            available=bool(payload.get("available", True)),
            source=str(payload.get("source", "offline_snapshot")),
            latest_document_type=str(payload.get("latest_document_type", "")),
            semantic_sentiment=float(payload.get("semantic_sentiment", 0.0)),
            execution_confidence=float(payload.get("execution_confidence", 0.0)),
            governance_red_flag=float(payload.get("governance_red_flag", 0.0)),
            key_phrases=[str(item) for item in payload.get("key_phrases", [])],
            key_risks=[str(item) for item in payload.get("key_risks", [])],
            notes=[str(item) for item in payload.get("notes", [])],
        )
        return self._store.save_semantic_snapshot(snapshot)
