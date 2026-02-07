import sqlite3


EVENTS_DB_PATH = ".db"
EVENTS_SUMMARY_DB_PATH = "events_summary.db"
# db_paths = [EVENTS_DB_PATH, EVENTS_SUMMARY_DB_PATH]
def init_db():
    conn = sqlite3.connect(EVENTS_DB_PATH,check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        session_id TEXT,
        device_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        sequence INTEGER,
        event_metadata TEXT,
        event_source TEXT CHECK(event_source IN ('client', 'server'))
    )
    """)
    conn.commit()
    return conn

def init_summaries_table():
    conn = sqlite3.connect(EVENTS_SUMMARY_DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events_summary (
        summary_id TEXT PRIMARY KEY,
        granularity TEXT NOT NULL,
        device_id TEXT,
        event_type TEXT NOT NULL,
        event_count INTEGER NOT NULL,
        event_source TEXT,
        bucket_start_ms INTEGER NOT NULL
    )
    """)
    conn.commit()
    return conn
    