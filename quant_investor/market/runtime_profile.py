"""Runtime stage profiling helpers for market analysis paths."""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping


@dataclass
class MarketRuntimeProfiler:
    """Collect stage timings for ``market analyze`` and ``market run``."""

    market: str
    universe: str
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    _started_at: float = field(
        default_factory=perf_counter,
        init=False,
        repr=False,
    )
    _stage_stack: list[dict[str, float]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @contextmanager
    def stage(
        self,
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        stage_metadata = dict(metadata or {})
        started_at = perf_counter()
        stack_frame = {"child_wall_seconds": 0.0}
        self._stage_stack.append(stack_frame)
        try:
            yield stage_metadata
        except Exception as exc:
            stage_metadata.setdefault("error", type(exc).__name__)
            raise
        finally:
            wall_seconds = perf_counter() - started_at
            if self._stage_stack and self._stage_stack[-1] is stack_frame:
                self._stage_stack.pop()
            child_wall_seconds = float(stack_frame.get("child_wall_seconds", 0.0))
            exclusive_seconds = max(0.0, wall_seconds - child_wall_seconds)
            if self._stage_stack:
                self._stage_stack[-1]["child_wall_seconds"] = (
                    float(self._stage_stack[-1].get("child_wall_seconds", 0.0))
                    + wall_seconds
                )
            self.stages.append(
                {
                    "name": str(name),
                    "seconds": exclusive_seconds,
                    "duration_ms": round(exclusive_seconds * 1000.0, 3),
                    "exclusive_seconds": exclusive_seconds,
                    "exclusive_duration_ms": round(exclusive_seconds * 1000.0, 3),
                    "wall_seconds": wall_seconds,
                    "wall_duration_ms": round(wall_seconds * 1000.0, 3),
                    "child_wall_seconds": child_wall_seconds,
                    "child_wall_duration_ms": round(child_wall_seconds * 1000.0, 3),
                    "metadata": stage_metadata,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        elapsed = perf_counter() - self._started_at
        return {
            "market": str(self.market),
            "universe": str(self.universe),
            "categories": list(self.categories),
            "metadata": dict(self.metadata),
            "total_seconds": elapsed,
            "total_duration_ms": round(elapsed * 1000.0, 3),
            "stages": list(self.stages),
        }

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return output_path

    def to_markdown(self) -> str:
        lines = [
            f"# Market Runtime Profile: {self.market} / {self.universe}",
            "",
            "| Stage | Exclusive Seconds | Wall Seconds | Metadata |",
            "| --- | ---: | ---: | --- |",
        ]
        for stage in self.stages:
            metadata = json.dumps(
                stage.get("metadata", {}),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            lines.append(
                "| {name} | {seconds:.3f} | {wall_seconds:.3f} | `{metadata}` |".format(
                    name=stage.get("name", ""),
                    seconds=float(stage.get("exclusive_seconds", stage.get("seconds", 0.0))),
                    wall_seconds=float(stage.get("wall_seconds", stage.get("seconds", 0.0))),
                    metadata=metadata,
                )
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def profile_stage(
    runtime_profiler: Any | None,
    name: str,
    metadata: Mapping[str, Any] | None = None,
):
    stage = getattr(runtime_profiler, "stage", None)
    if callable(stage):
        return stage(name, metadata or {})
    return nullcontext(dict(metadata or {}))


__all__ = ["MarketRuntimeProfiler", "profile_stage"]
