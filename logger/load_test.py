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

    for j in range(BATCH_SIZE):
        event = (
            str(uuid.uuid4()),
            "pi-test",
            "BARK",
            int(time.time() * 1000),
            i + j,
            json.dumps({"amp": 0.73})
        )
        cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", event)

    conn.commit()

    batch_end = time.time()
    latencies.append((batch_end - batch_start) * 1000)

    if i % 5000 == 0:
        print(f"Inserted {i} events")
print("Done.")
print(f"Avg batch latency (ms): {sum(latencies)/len(latencies):.2f}")
print(f"Max batch latency (ms): {max(latencies):.2f}")