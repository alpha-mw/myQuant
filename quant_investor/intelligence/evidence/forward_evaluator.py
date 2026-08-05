"""Sprint R2.2 extension seam; I0 deliberately provides no evaluator implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ForwardEvidenceEvaluator(Protocol):
    """Read-only interface for a future R2.2 evidence evaluation implementation."""

    def evaluate(
        self,
        *,
        observation_bundle: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        matured_label_refs: Sequence[Mapping[str, Any]],
        evaluation_receipt_refs: Sequence[Mapping[str, Any]],
        as_of: str,
    ) -> Mapping[str, Any]: ...


__all__ = ["ForwardEvidenceEvaluator"]
