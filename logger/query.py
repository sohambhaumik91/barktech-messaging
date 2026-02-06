import sqlite3
DB_PATH = "events.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# cursor.execute("CREATE INDEX idx_events_device_type_time on events(device_id, event_type, timestamp_ms)")
# cursor.execute("DROP INDEX idx_events_device_type_time")
cursor.execute("DROP TABLE events")
conn.commit()
