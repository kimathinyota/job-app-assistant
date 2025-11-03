# run.py
import os
import uvicorn
import logging

# --- THIS IS THE CRITICAL FIX ---
# Set these environment variables BEFORE anything else is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging to see the effect
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("ENVIRONMENT VARIABLES SET. Starting Uvicorn...")
log.info(f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}")
log.info(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")


if __name__ == "__main__":
    # Now we can safely run uvicorn.
    # We run it as a function call instead of a shell command.
    uvicorn.run(
        "backend.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False  # <--- THIS IS THE FIX.
    )