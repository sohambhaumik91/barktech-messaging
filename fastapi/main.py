from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import os
from pydantic import BaseModel
from typing import List, Optional, Union
import threading
import win32pipe
import win32file
import pywintypes
import json
import subprocess
from queue import Queue
import time



PIPE_NAME = r"\\.\pipe\ffmpeg_pipe"
BUFFER_SIZE = 64 * 1024
STREAM_ENDPOINT = "srt://127.0.0.1:9000?mode=listener&latency=20"

class VideStreamListener:
    def __init__(self):
        self.stream_cmd =  ["ffmpeg", 
                    "-fflags", "nobuffer"
                    , "-flags", "low_delay" 
                    , "-probesize", "32" 
                    , "-i", STREAM_ENDPOINT 
                    , "-map", "0:v:0" 
                    , "-pix_fmt", "bgr24" 
                    , "-vcodec", "rawvideo" 
                    , "-an", "-sn" 
                    , "-f", "rawvideo" 
                    , f"{PIPE_NAME}"]        
        self.running = False
        self.stream_listener = None
        self.SUCCESS_INDICATOR_TOKEN = "Output #0"
        
    def check_stream_health(self, queue_scheduler):
        for line in iter(self.stream_listener.stderr.readline, ''):
            print("FFmpeg receiver:", line.decode(errors='ignore').strip())
            err_line = line.strip()
            if self.SUCCESS_INDICATOR_TOKEN in err_line:
                queue_scheduler.put("SUCCESS")
            
    def kill_stream(self):
        if self.stream_listener and self.stream_listener.poll() is None:
            self.stream_listener.kill()
            self.stream_listener.stderr.close()
    def start_stream(self):
        try: 
            self.stream_listener = subprocess.Popen(self.stream_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
            self.running = True
        except Exception as err:
            print(f"An error occurred at the time of starting the stream: {err}")
    
        

class PipeReader:
    def __init__(self):
        self.latest = None
        self.lock = threading.Lock()
        self.running = False
        self.total_bytes = 0
        
    def create_pipe(self):
        return win32pipe.CreateNamedPipe(
            PIPE_NAME,
            win32pipe.PIPE_ACCESS_INBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
            1,  
            0,
            BUFFER_SIZE * 16,
            0,
            None,
        )
        
    def run(self):
        self.running = True
        pipe = self.create_pipe()
        
        try:
            print("Waiting for FFmpeg to connect to named pipe...")
            win32pipe.ConnectNamedPipe(pipe, None)
            print("FFmpeg connected! Starting to read data...")
            while self.running:
                try:
                    hr, data = win32file.ReadFile(pipe, BUFFER_SIZE)
                    if data:
                        self.total_bytes += len(data)
                        with self.lock:
                            self.latest = data
                        if self.total_bytes % (10 * 1024 * 1024) < BUFFER_SIZE:
                            print(f"Received {self.total_bytes / (1024*1024):.2f} MB")
                except pywintypes.error as e:
                    if e.winerror == 109:
                        print("FFmpeg disconnected (pipe broken)")
                        break
                    else:
                        print(f"Pipe read error: {e}")
                        break
                        
        finally:
            print(f"Closing pipe. Total bytes received: {self.total_bytes}")
            win32file.CloseHandle(pipe)
            
    def stop(self):
        self.running = False

class BarkTechServer_v2():
    def __init__(self):
        self.app = FastAPI()
        self._setup_middleware()
        self.state = {
            "bytes_received": 0
        }
        self.stream_reader = PipeReader()
        self.reader_thread = None
        self.stream_receiver = VideStreamListener()
        self.video_stream_queue = Queue()
        
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    async def setup_routes(self):
        @self.app.on_event("startup")
        def start_pipe_thread():
            print("Starting pipe reader thread...")
            self.reader_thread = threading.Thread(target=self.stream_reader.run, daemon=True)
            self.reader_thread.start()
            
        @self.app.on_event("shutdown")
        def stop_pipe_thread():
            print("Stopping pipe reader...")
            self.stream_reader.stop()
                    
        @self.app.get("/healthcheck", status_code=status.HTTP_200_OK)
        async def get_server_health():
            return {
                "health": "ok",
                "bytes_received": self.stream_reader.total_bytes
            }
        
        @self.app.get("/latest_frame_size")
        async def get_latest_frame_size():
            with self.stream_reader.lock:
                if self.stream_reader.latest:
                    return {"size": len(self.stream_reader.latest)}
                return {"size": 0}
        
        @self.app.post("/init_stream_check", status_code=status.HTTP_200_OK)
        async def handle_stream_init_request(req_body):
            try:
                print(f"request body: {json.dumps(req_body)}")
                self.stream_receiver.start_stream()
                err_reader = threading.Thread(target = self.stream_receiver.check_stream_health, args= (self.video_stream_queue))
                await asyncio.sleep(1)
                err_reader.start()
                start = time.time()
                while time.time() - start < 3:
                    err_message = self.video_stream_queue.get(timeout=2)
                    if err_message == "SUCCESS":
                        return {"status" : "ffmpeg started and healthy", "health" : "ok"}
                    elif "ERROR" in err_message or "Invalid" in err_message or "No such file" in err_message:
                        self.stream_receiver.kill_stream()
                        return {"status" : "ffmpeg failed to start. check server logs", "health" : "not-ok"}
                err_reader.join()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )
            
    async def start(self):
        await self.setup_routes()
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config=config)
        await server.serve()

if __name__ == "__main__":
    bt_server = BarkTechServer_v2()
    try:
        asyncio.run(bt_server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")