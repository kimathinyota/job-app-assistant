from backend.core.registry import Registry
import logging

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# This is the single, shared instance of the registry.
# It's created once when the server starts.
db_path = "./backend/data/db.json"
log.info(f"Initializing singleton Registry with path: {db_path}")

# This 'registry' instance is imported directly by all route files
# to prevent race conditions and database corruption.
registry = Registry(db_path)

