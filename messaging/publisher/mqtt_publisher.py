import argparse
import uuid
import time
import paho.mqtt.client as mqtt
import json

parser = argparse.ArgumentParser("Accept variables to pass connections to connect to broker server")
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, default=1883)
parser.add_argument("--topic", required=True)
parser.add_argument("--user", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--device-id", required=True)
args = parser.parse_args()

def build_event(device_id: str):
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "HEARTBEAT",
        "source": "device",
        "device_id": device_id,
        "region": "IN",
        "timestamp": int(time.time()),
        "metadata": {
            "status": "alive"
        }
    }
    
def main():
    client = mqtt.Client()
    client.username_pw_set(args.user, args.password)

    print(f"Connecting to MQTT broker at {args.host}:{args.port}")
    client.connect(args.host, args.port, keepalive=60)

    event = build_event(args.device_id)
    payload = json.dumps(event)

    print(f"Publishing to topic: {args.topic}")
    publish_info  = client.publish(args.topic, payload, qos=1)
    publish_info.wait_for_publish()
    client.disconnect()


if __name__ == "__main__":
    main()