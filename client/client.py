import uuid
import aiohttp
import asyncio
import subprocess
from pathlib import Path

## write code for device to load id from local config and fetch streaming configs from the server. maybe AUTH if required!
class Client:
    def __init__(self):
        self.server_base_url = "http://127.0.0.1:8000"
        self.device_id = uuid.uuid4()
        self.stream_cmd = [
            "ffmpeg", "-re", "-stream_loop", "-1",  "-i", str(Path.cwd() / "fsod.mp4"), "-c:v", "libx264",  "-preset", "veryfast",  "-tune", "zerolatency",  "-pix_fmt", "yuv420p",  "-f" "mpegts",  "srt://127.0.0.1:9000?mode=caller&latency=20"
        ]
        self.sender_proc = None
        self.stream_is_running = False
    
    async def client_stream_start_ack(self):
        payload = {
            "device_id": uuid.uuid4(),
            "codec": "H.264",
            "resolution": "480 * 480",
            "fps": 30
        }
        stream_endpoint = f"{self.server_base_url}/init_stream_check"
        async with aiohttp.ClientSession() as session:
            async with session.post(stream_endpoint, json=payload) as resp:
                data = await resp.json()
                print(data)
                if data.status == "ok":
                    #start_client streem
                    self.start_ffmpeg_stream()
                else:
                    print("server listener failed to start")
                    # maybe implement a number of retries before giving up
                ## based on response received from the server, it will start an ffmpeg stream
    
    def start_ffmpeg_stream(self):
        try:
            ##TO_DO add health check on the client stream
            self.sender_proc  = subprocess.Popen(self.stream_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            raise Exception("There was an error in starting the stream from the client")

        
    
    def stop_ffmpeg_stream(self):
        pass
    
            
if __name__ == "__main__":
    client = Client()
    asyncio.run(client.client_stream_start_ack())


