import json
import argparse
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected to broker with result code:", rc, userdata)
    
    client.subscribe(userdata["topic"], qos=1)

def on_message(client, userdata, msg):
    print("\n--- EVENT RECEIVED ---")
    print("Topic:", msg.topic)

    try:
        payload = json.loads(msg.payload.decode())
        print("Payload:", json.dumps(payload, indent=2))
    except json.JSONDecodeError:
        print("Invalid JSON:", msg.payload)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)

    args = parser.parse_args()

    client = mqtt.Client(userdata={"topic": args.topic})
    client.username_pw_set(args.user, args.password)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {args.host}:{args.port}")
    client.connect(args.host, args.port, keepalive=60)

    print("Waiting for messages...")
    client.loop_forever()

if __name__ == "__main__":
    main()
