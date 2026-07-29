"""Side-effect-free immutable storage for typed industry evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class IndustryEvidence:
    """One caller-supplied evidence reference; no source is fetched here."""

    industry_id: str
    evidence_ref: str
    evidence_type: str
    available_at: str
    summary: str

    def __post_init__(self) -> None:
        for field_name in (
            "industry_id",
            "evidence_ref",
            "evidence_type",
            "available_at",
            "summary",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class IndustryEvidenceStore:
    """An immutable in-memory evidence collection."""

    evidence: tuple[IndustryEvidence, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(self.evidence))
        identities = tuple((row.industry_id, row.evidence_ref) for row in normalized)
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate industry evidence reference")
        object.__setattr__(self, "evidence", normalized)

    def add(self, evidence: IndustryEvidence) -> "IndustryEvidenceStore":
        if not isinstance(evidence, IndustryEvidence):
            raise ValueError("evidence must be IndustryEvidence")
        return IndustryEvidenceStore(self.evidence + (evidence,))

    def for_industry(self, industry_id: str) -> tuple[IndustryEvidence, ...]:
        if not isinstance(industry_id, str) or not industry_id:
            raise ValueError("industry_id must be a non-empty string")
        return tuple(row for row in self.evidence if row.industry_id == industry_id)


__all__ = ["IndustryEvidence", "IndustryEvidenceStore"]
