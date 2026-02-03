# MediaMTX Setup Guide for Raspberry Pi

This README explains how to run **MediaMTX** as the media server for this project on a Raspberry Pi, and how to connect your existing FFmpeg / inference pipeline to it.

The goal is simple:

> **Inference / camera → FFmpeg → MediaMTX → Web / Clients**

---

## 1. What MediaMTX Does in This Project

MediaMTX acts as a **central media hub**:

- Accepts video streams from FFmpeg (RTSP / WebRTC / HLS / MJPEG)
- Re-publishes them to browsers or other clients
- Removes the need for custom pipes or ad-hoc streaming servers

You should use MediaMTX when:
- Multiple consumers need the same stream
- You want browser-friendly playback (WebRTC / HLS)
- You want a clean boundary between *producing* and *consuming* video

---

## 2. Supported Raspberry Pi Models

- Raspberry Pi 4 (recommended)
- Raspberry Pi 3 (works, lower performance)

OS tested:
- Debian Bookworm (32-bit preferred)

---

## 3. Install MediaMTX

### 3.1 Download Binary

```bash
cd ~
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_arm64.tar.gz
```

For 32-bit OS, use:

```bash
mediamtx_linux_armv7.tar.gz
```

### 3.2 Extract

```bash
tar -xzf mediamtx_linux_arm64.tar.gz
cd mediamtx
```

### 3.3 Verify

```bash
./mediamtx --version
```

---

## 4. MediaMTX Configuration

The default config works for most cases.

Edit the config:

```bash
nano mediamtx.yml
```

Minimum recommended settings:

```yaml
rtsp: yes
webrtc: yes
hls: yes

paths:
  all:
    source: publisher
```
Uncomment the ICE server url.
Save and exit.

---

## 5. Running MediaMTX

### 5.1 Run Manually (for testing)

```bash
./mediamtx mediamtx.yml
```

You should see logs like:

```
RTSP server listening on :8554
WebRTC server listening on :8889
HLS server listening on :8888
```

---

## 6. Streaming **Into** MediaMTX Using FFmpeg

### 6.1 Example: RTSP Publish

```bash
rpicam-vid -t 0 --inline --nopreview -o - | ffmpeg -i - -c:v copy -f rtsp rtsp://localhost:8554/mystream
```

This publishes a stream named `mystream`.

---


## 7. Viewing the Stream
# check the code in /raspberrpi-server/test-client
---

## 8. Running MediaMTX as a Service (Recommended)

### 8.1 Create Systemd Service

```bash
sudo nano /etc/systemd/system/mediamtx.service
```

```ini
[Unit]
Description=MediaMTX Server
After=network.target

[Service]
ExecStart=/home/pi/mediamtx/mediamtx /home/pi/mediamtx/mediamtx.yml
WorkingDirectory=/home/pi/mediamtx
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

### 8.2 Enable

```bash
sudo systemctl daemon-reload
sudo systemctl enable mediamtx
sudo systemctl start mediamtx
```

Check status:

```bash
sudo systemctl status mediamtx
```

---

## 9. Common Issues

### Stream Not Visible
- Ensure FFmpeg is publishing successfully
- Check MediaMTX logs

### Browser Can’t Load WebRTC
- Make sure ports **8889 UDP/TCP** are open
- If behind NGINX, WebRTC must be proxied carefully (or bypassed)

### High Latency
- Use WebRTC instead of HLS
- Use `-preset ultrafast` and `-tune zerolatency`

---

## 10. Architecture Summary

```
Camera / Inference
       ↓
     FFmpeg
       ↓
    MediaMTX
   ↓     ↓
 WebRTC  RTSP/HLS
```

MediaMTX becomes the **single source of truth** for video streaming.

---

## 11. Next Steps

- Add authentication to MediaMTX
- Enable recording
- Add multiple camera paths
- Proxy MediaMTX through NGINX

---

If you’re extending this project later (cloud relay, mobile app, multi-cam), MediaMTX will scale with minimal changes.
