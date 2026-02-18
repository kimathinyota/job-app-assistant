# backend/worker_inference.py
import sys
import os
from redis import Redis
from rq import SimpleWorker, Queue 

sys.path.append(os.getcwd())
from backend.tasks import initialize_inference_worker

# --- PRIORITY CONFIGURATION ---
# The order here MATTERS. 
# 1. 'q_inference': High Priority (User clicked a button, Page load)
# 2. 'q_background': Low Priority (Bulk updates, CV auto-rescore)
listen = ['q_inference', 'q_background'] 

redis_url = 'redis://localhost:6379' 
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    print(f"🐇 FAST WORKER STARTED")
    print(f"   - Priority Queues: {listen}")
    
    initialize_inference_worker()

    worker = SimpleWorker([Queue(name, connection=conn) for name in listen], connection=conn)
    worker.work()