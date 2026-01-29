import uuid
import aiohttp
import asyncio
import subprocess
from pathlib import Path
import time
import paho.mqtt.client as mqtt
import json

## write code for device to load id from local config and fetch streaming configs from the server. maybe AUTH if required!
class Client:
    def __init__(self):
        self.server_base_url = "http://127.0.0.1:8000"
        self.device_id = uuid.uuid4()
        self.stream_cmd = [
            "ffmpeg", "-re", "-stream_loop", "-1",  "-i", str(Path.cwd() / "fsod.mp4"), "-c:v", "libx264", "-vf", "scale=480:480" , "-preset", "veryfast",  "-tune", "zerolatency",  "-pix_fmt", "yuv420p",  "-f", "mpegts",  "srt://127.0.0.1:9000?mode=caller"
        ]
        self.sender_proc = None
        self.stream_is_running = False
        self.mqtt_client = None
    
    async def client_stream_start_ack(self):
        payload = {
            "device_id": str(uuid.uuid4()),
            "codec": "H.264",
            "resolution": "480 * 480",
            "fps": "30"
        }
        stream_endpoint = f"{self.server_base_url}/init_stream_check"
        print(f"stream endpoint is : {stream_endpoint}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(stream_endpoint, json=payload) as resp:
                    if resp.status != 200:
                        print("Server error:", await resp.text())
                        return
                    data = await resp.json()
                    print(data["health"])
                    if data["health"].strip() == "ok":
                        self.start_ffmpeg_stream()
                    else:
                        print("server listener failed to start")
        except Exception as err:
            print(f"An exception occurred during startup: {err}")
    
    def start_ffmpeg_stream(self):
        try:
            print("starting stream from client ....")
            ##TO_DO add health check on the client stream
            self.sender_proc  = subprocess.Popen(self.stream_cmd, stdout=None, stderr=None)
        except:
            raise Exception("There was an error in starting the stream from the client")
    @staticmethod
    def client_on_connect(client, userdata, flags, rc):
        print("Connected to broker with result code:", rc, userdata)
    
        client.subscribe(userdata["topic"], qos=1)
    
    @staticmethod
    def client_on_message(client, userdata, msg):
        print("\n--- EVENT RECEIVED ---")
        print("Topic:", msg.topic)

        try:
            payload = json.loads(msg.payload.decode())
            print("Payload:", json.dumps(payload, indent=2))
        except json.JSONDecodeError:
            print("Invalid JSON:", msg.payload)
    
    def subscribe_to_mqtt(self):
        self.mqtt_client = mqtt.Client(userdata={"topic": "barktech/inference"})
        self.mqtt_client.username_pw_set("guest", "guest")
        self.mqtt_client.on_connect = Client.client_on_connect
        self.mqtt_client.on_message = Client.client_on_message
        self.mqtt_client.connect("127.0.0.1", 1883, keepalive=60)
                
    def stop_ffmpeg_stream(self):
        if self.sender_proc:
            self.sender_proc.terminate()
            self.sender_proc.wait()
            self.stream_is_running = False
    
            
if __name__ == "__main__":
    client = Client()
    try: 
        asyncio.run(client.client_stream_start_ack())
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        client.stop_ffmpeg_stream()
    except Exception as e:
        print("An error occurred when client was trying to start up.")


