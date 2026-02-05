import sqlite3


DB_PATH = "events.db"
def init_db():
    conn = sqlite3.connect(DB_PATH,check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        device_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        sequence INTEGER,
        payload_json TEXT
    )
    """)
    conn.commit()
    return conn