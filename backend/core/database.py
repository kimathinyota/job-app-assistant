# backend/core/database.py
from tinydb import TinyDB, Query
from pathlib import Path
from typing import Any, Dict, List, Optional
import json 
import threading # <-- Import the threading library
import logging

log = logging.getLogger(__name__)

class TinyDBManager:
    def __init__(self, path: str = "./backend/data/db.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = TinyDB(path)
        log.info(f"TinyDBManager initialized. Creating threading.Lock().")
        self.lock = threading.Lock() # <-- This is now safe

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