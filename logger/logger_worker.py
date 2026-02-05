import json
import time
import threading
from queue import Queue, Empty, Full
import paho.mqtt.client as mqtt
import sqlite3

class LoggerService:
    def __init__(self, db_path, batch_size=100):
        self.db_path = db_path
        self.batch_size = batch_size

        self.queue = Queue(maxsize=10_000)
        self.flush_interval_s = 0.5
        self.queue_poll_timeout_s = 0.1

        self.dropped_events = 0
        self.running = True

        self.mqtt_client = None

    # ---------- MQTT CALLBACKS ----------

    def client_on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            try:
                self.queue.put_nowait(payload)
            except Full:
                self.dropped_events += 1
        except json.JSONDecodeError:
            pass  # ignore malformed payloads

    @staticmethod
    def client_on_connect(client, userdata, flags, rc):
        client.subscribe(userdata["topic"], qos=1)

    
    def event_batcher(self):
        batch = []
        last_flush = time.monotonic()
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None  # explicit transactions
        )
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        while self.running:
            try:
                event = self.queue.get(timeout=self.queue_poll_timeout_s)
                batch.append(event)
            except Empty:
                pass

            now = time.monotonic()

            if (
                len(batch) >= self.batch_size
                or (batch and now - last_flush >= self.flush_interval_s)
            ):
                self.write_batch_to_db(batch)
                batch.clear()
                last_flush = now


    def write_batch_to_db(self, cursor, conn, event_batch):
        try:
            cursor.execute("BEGIN")
            cursor.executemany(
                """
                INSERT INTO events
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                event_batch
            )
            conn.commit()
            print(f"Flushed {len(event_batch)} events (dropped={self.dropped_events})")
        except Exception as e:
            conn.rollback()
            print(f"DB error: {e}")


    def start(self):
        threading.Thread(target=self.event_batcher, daemon=True).start()

        self.mqtt_client = mqtt.Client(userdata={"topic": "barktech/inference"})
        self.mqtt_client.on_connect = LoggerService.client_on_connect
        self.mqtt_client.on_message = self.client_on_message
        self.mqtt_client.connect("127.0.0.1", 1883, keepalive=60)
        self.mqtt_client.loop_forever()
