import time
from db import init_db

conn = init_db()
cur = conn.cursor()

device_id = "pi-test"
now = int(time.time() * 1000)
window_ms = 10 * 60 * 1000  # last 10 minutes

latencies = []

print("Running read test...")

for i in range(10):
    start = time.time()

    cur.execute("""
    SELECT COUNT(*)
    FROM events
    WHERE device_id = ?
      AND event_type = 'BARK'
      AND timestamp_ms > ?
    """, (device_id, now - window_ms))

    count = cur.fetchone()[0]
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)

    print(f"Run {i}: {round(elapsed, 2)} ms, count={count}")

print("----")
print(f"Avg read latency (ms): {sum(latencies)/len(latencies):.2f}")
print(f"Max read latency (ms): {max(latencies):.2f}")
