# backend/core/dependencies.py
from backend.core.registry import Registry
from backend.core.inferer import MappingInferer
import logging

log = logging.getLogger(__name__)

db_path = "./backend/data/db.json"
log.info(f"Initializing singleton Registry with path: {db_path}")

# This is safe again. The lock will be created in the reloader
# BUT it won't conflict because the NLP threads are disabled.
registry = Registry(db_path)

log.info("Initializing singleton MappingInferer (models will be loaded at startup)...")
# This is safe. It just calls the lightweight __init__.
inferer = MappingInferer()