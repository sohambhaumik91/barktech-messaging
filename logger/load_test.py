from db import init_db
import time
import uuid
import json


TOTAL_EVENTS = 50000
BATCH_SIZE = 100
conn = init_db()
cur = conn.cursor()
latencies = []

for i in range(0, TOTAL_EVENTS, BATCH_SIZE):
    batch_start = time.time()

    batch = []
    for j in range(BATCH_SIZE):
        batch.append((
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "pi-test",
            "BARK",
            int(time.time() * 1000),
            i + j,
            json.dumps({"amp": 0.73}),
            "client"
        ))
    try:
        cur.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()

        batch_end = time.time()
        latencies.append((batch_end - batch_start) * 1000)
    except Exception as e:
        conn.rollback()
        print(f"DB error at batch starting index {i}: {e}")
        # optional: break if you want to stop on first failure
print("Done.")
print(f"Avg batch latency (ms): {sum(latencies)/len(latencies):.2f}")
print(f"Max batch latency (ms): {max(latencies):.2f}")