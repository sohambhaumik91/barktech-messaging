from fastapi import FastAPI, status
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

PIPE_NAME = r"\\.\pipe\ffmpeg_pipe"
BUFFER_SIZE = 64 * 1024  # Read in 64KB chunks

class PipeReader:
    def __init__(self):
        self.latest = None
        self.lock = threading.Lock()
        self.running = False
        self.total_bytes = 0
        
    def create_pipe(self):
        """Create the named pipe"""
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
                    if e.winerror == 109:  # ERROR_BROKEN_PIPE
                        print("FFmpeg disconnected (pipe broken)")
                        break
                    elif e.winerror == 232:  # ERROR_NO_DATA
                        print("No more data available")
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