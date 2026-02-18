import sys
import os
from redis import Redis
from rq import SimpleWorker, Queue 

sys.path.append(os.getcwd())
from backend.tasks import initialize_parsing_worker

# This worker listens to the "Slow Lane"
listen = ['q_parsing']
redis_url = 'redis://localhost:6379' 
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    print(f"🐢 PARSING WORKER STARTED")
    print(f"   - Listening on: {listen}")
    
    # Load ONLY Llama 3 (6GB)
    initialize_parsing_worker()

    worker = SimpleWorker([Queue(name, connection=conn) for name in listen], connection=conn)
    worker.work()