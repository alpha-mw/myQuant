"""SQLite-backed preset CRUD using the shared web_runs.db."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from web.services.run_history_store import history_store


class PresetStore:
    """CRUD operations for research parameter presets."""

    def _conn(self):
        return history_store._conn()

    def list_presets(self) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM presets ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["config"] = json.loads(item.pop("config_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                item["config"] = {}
            result.append(item)
        return result

    def get_preset(self, preset_id: str) -> Optional[dict[str, Any]]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM presets WHERE preset_id = ?", (preset_id,)
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["config"] = json.loads(item.pop("config_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            item["config"] = {}
        return item

    def create_preset(
        self, name: str, description: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        preset_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO presets (preset_id, name, description, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (preset_id, name, description, json.dumps(config), now, now),
        )
        conn.commit()
        return {
            "preset_id": preset_id,
            "name": name,
            "description": description,
            "config": config,
            "created_at": now,
            "updated_at": now,
        }

    def update_preset(
        self,
        preset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        existing = self.get_preset(preset_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_name = name if name is not None else existing["name"]
        new_desc = description if description is not None else existing["description"]
        new_config = config if config is not None else existing["config"]

        conn = self._conn()
        conn.execute(
            """
            UPDATE presets SET name = ?, description = ?, config_json = ?, updated_at = ?
            WHERE preset_id = ?
            """,
            (new_name, new_desc, json.dumps(new_config), now, preset_id),
        )
        conn.commit()
        return {
            "preset_id": preset_id,
            "name": new_name,
            "description": new_desc,
            "config": new_config,
            "created_at": existing["created_at"],
            "updated_at": now,
        }

    def delete_preset(self, preset_id: str) -> bool:
        conn = self._conn()
        cursor = conn.execute(
            "DELETE FROM presets WHERE preset_id = ?", (preset_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


preset_store = PresetStore()
