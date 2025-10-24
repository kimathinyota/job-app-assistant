from tinydb import TinyDB, Query
from pathlib import Path
from typing import Any, Dict, List, Optional


class TinyDBManager:
    def __init__(self, path: str = "./backend/data/db.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = TinyDB(path)

    def insert(self, table: str, obj):
        tbl = self.db.table(table)
        tbl.upsert(obj.model_dump(), Query().id == obj.id)

    def get(self, table: str, obj_id: str) -> Optional[Dict[str, Any]]:
        return self.db.table(table).get(Query().id == obj_id)

    def all(self, table: str) -> List[Dict[str, Any]]:
        return self.db.table(table).all()

    def remove(self, table: str, obj_id: str):
        self.db.table(table).remove(Query().id == obj_id)
