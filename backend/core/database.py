from tinydb import TinyDB, Query
from pathlib import Path
from typing import Any, Dict, List, Optional
import json # <-- NEW: Import the standard json library


class TinyDBManager:
    def __init__(self, path: str = "./backend/data/db.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Note: In a production environment, you might configure TinyDB with a custom JSON encoder here,
        # but the Pydantic approach is cleaner for this project.
        self.db = TinyDB(path)

    def insert(self, table: str, obj):
        tbl = self.db.table(table)
        
        # FIX: Use model_dump_json() to serialize all fields (including datetime) to a JSON string,
        # then immediately loads() it back into a Python dictionary containing JSON-safe strings.
        data_to_insert = json.loads(obj.model_dump_json())
        
        tbl.upsert(data_to_insert, Query().id == obj.id)

    def get(self, table: str, obj_id: str) -> Optional[Dict[str, Any]]:
        return self.db.table(table).get(Query().id == obj_id)

    def all(self, table: str) -> List[Dict[str, Any]]:
        return self.db.table(table).all()

    def remove(self, table: str, obj_id: str):
        self.db.table(table).remove(Query().id == obj_id)

