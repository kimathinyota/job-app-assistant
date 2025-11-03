# backend/core/database.py
from tinydb import TinyDB, Query
from pathlib import Path
from typing import Any, Dict, List, Optional
import json 
import threading # <-- Import the threading library

class TinyDBManager:
    def __init__(self, path: str = "./backend/data/db.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = TinyDB(path)
        self.lock = threading.Lock() # <-- This is fine now

    def insert(self, table: str, obj):
        with self.lock:
            data_to_insert = json.loads(obj.model_dump_json())
            tbl = self.db.table(table)
            tbl.upsert(data_to_insert, Query().id == obj.id)

    def get(self, table: str, obj_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.db.table(table).get(Query().id == obj_id)

    def all(self, table: str) -> List[Dict[str, Any]]:
        with self.lock:
            return self.db.table(table).all()

    def remove(self, table: str, obj_id: str):
        with self.lock:
            self.db.table(table).remove(Query().id == obj_id)