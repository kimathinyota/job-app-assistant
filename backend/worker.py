# backend/worker.py
import sys
import os
from redis import Redis
# --- CHANGE 1: Import SimpleWorker instead of Worker ---
from rq import SimpleWorker, Queue 

# Add project root to path so backend.tasks can be found
sys.path.append(os.getcwd())

# Import the loader function
from backend.tasks import initialize_worker

# Configuration
listen = ['default']
redis_url = 'redis://localhost:6379' 

# Create the Redis connection
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    print(f"🚀 Starting Metal-Optimized Worker connected to {redis_url}")
    
    # PRE-LOAD: Load the model into Mac RAM/GPU
    initialize_worker()

    # 1. Instantiate Queues
    queues = [Queue(name, connection=conn) for name in listen]

    # --- CHANGE 2: Use SimpleWorker ---
    # SimpleWorker runs the job in the *current* process (Main Thread).
    # This avoids fork(), keeping the Mac GPU context alive and safe.
    worker = SimpleWorker(queues, connection=conn)
    
    # 3. Start working
    worker.work()