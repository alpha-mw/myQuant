from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "theme_taxonomy.v2"
DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("data") / "theme_taxonomy.v2.json"


@dataclass(frozen=True)
class ThemeTaxonomyNode:
    theme_id: str
    name: str
    parent_id: str = ""
    aliases: tuple[str, ...] = ()
    mandate: str = "observation"
    tradable_node: bool = False
    supply_chain_roles: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ThemeTaxonomyNode":
        theme_id = str(payload.get("theme_id") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not theme_id or "::" not in theme_id:
            raise ValueError("taxonomy theme_id must be namespaced")
        if not name:
            raise ValueError(f"taxonomy node {theme_id} requires name")
        mandate = str(payload.get("mandate") or "observation").strip().lower()
        if mandate not in {"technology", "advanced_manufacturing", "tactical", "observation"}:
            raise ValueError(f"taxonomy node {theme_id} has invalid mandate={mandate}")
        return cls(
            theme_id=theme_id,
            name=name,
            parent_id=str(payload.get("parent_id") or "").strip(),
            aliases=tuple(_texts(payload.get("aliases"))),
            mandate=mandate,
            tradable_node=payload.get("tradable_node") is True,
            supply_chain_roles=tuple(_texts(payload.get("supply_chain_roles"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "aliases": list(self.aliases),
            "mandate": self.mandate,
            "tradable_node": self.tradable_node,
            "supply_chain_roles": list(self.supply_chain_roles),
        }


@dataclass(frozen=True)
class ThemeTaxonomy:
    nodes: tuple[ThemeTaxonomyNode, ...]
    schema_version: str = SCHEMA_VERSION
    taxonomy_id: str = "myquant-tech-v2"
    version: str = "2.0.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ThemeTaxonomy":
        schema_version = str(payload.get("schema_version") or "").strip()
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={schema_version}")
        raw_nodes = payload.get("nodes")
        if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
            raise ValueError("taxonomy nodes must be a list")
        nodes = tuple(
            ThemeTaxonomyNode.from_mapping(item)
            for item in raw_nodes
            if isinstance(item, Mapping)
        )
        if not nodes:
            raise ValueError("taxonomy requires at least one node")
        by_id = {node.theme_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("taxonomy theme_id values must be unique")
        for node in nodes:
            if node.parent_id and node.parent_id not in by_id:
                raise ValueError(f"taxonomy parent missing for {node.theme_id}: {node.parent_id}")
        alias_owner: dict[str, str] = {}
        for node in nodes:
            for alias in (node.name, *node.aliases):
                key = _alias_key(alias)
                owner = alias_owner.setdefault(key, node.theme_id)
                if owner != node.theme_id:
                    raise ValueError(f"taxonomy alias collision: {alias}")
        return cls(
            nodes=nodes,
            schema_version=schema_version,
            taxonomy_id=str(payload.get("taxonomy_id") or "myquant-tech-v2"),
            version=str(payload.get("version") or "2.0.0"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ThemeTaxonomy":
        source = Path(path) if path else DEFAULT_TAXONOMY_PATH
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("taxonomy file must contain an object")
        return cls.from_mapping(payload)

    def node(self, theme_id: str) -> ThemeTaxonomyNode | None:
        return next((node for node in self.nodes if node.theme_id == theme_id), None)

    def resolve(self, value: str) -> ThemeTaxonomyNode | None:
        text = str(value or "").strip()
        direct = self.node(text)
        if direct is not None:
            return direct
        key = _alias_key(text)
        return next(
            (
                node
                for node in self.nodes
                if key in {_alias_key(node.name), *(_alias_key(alias) for alias in node.aliases)}
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taxonomy_id": self.taxonomy_id,
            "version": self.version,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": dict(self.metadata),
        }


def _texts(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in list(value or []):
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _alias_key(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())
