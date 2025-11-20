# run.py
import os
import uvicorn
import logging
import multiprocessing # <-- 1. Import multiprocessing

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
log.info(f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}")


if __name__ == "__main__":
    # --- 2. THIS IS THE FIX ---
    # Set the start method to 'spawn' BEFORE uvicorn runs.
    # This creates a clean new process instead of 'forking' the main one,
    # which avoids all threading/mutex inheritance issues.
    try:
        multiprocessing.set_start_method("spawn", force=True)
        log.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        log.warning("Could not set 'spawn' start method (might be already set).")
    # --- END OF FIX ---

    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False  # Must be False
    )