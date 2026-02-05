from db import init_db
import time
import uuid
import json

conn = init_db()
cur = conn.cursor()

event = {
    "event_id": str(uuid.uuid4()),
    "device_id": "pi-test",
    "event_type": "BOOT",
    "timestamp_ms": int(time.time() * 1000),
    "sequence": 1,
    "payload_json": json.dumps({"reason": "cold_start"})
}

cur.execute("""
INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)
""", tuple(event.values()))

conn.commit()

print("Inserted 1 event")
