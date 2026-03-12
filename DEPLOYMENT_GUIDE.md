"""
DEPLOYMENT GUIDE: Production-Ready Football Commentary System
Covers: Cloud, Edge, Docker, Broadcasting, and Real-time Streaming
"""

# ============================================================================
# 1. LOCAL DEPLOYMENT (Desktop/Laptop)
# ============================================================================

"""
Simplest deployment - runs on your machine
"""

# Step 1: Install (see REQUIREMENTS_AND_SETUP.md)
pip install -r requirements.txt
ollama run mistral  # In background

# Step 2: Run system
python full_system.py "football_match.mp4" ollama

# Configuration options (edit full_system.py):
backend_choice = "ollama"      # "ollama", "openai", "huggingface"
video_source = "match.mp4"     # File or webcam (0)
home_team = "Manchester Utd"
away_team = "Liverpool"
enable_voice = True

# Performance tuning for local machines (4GB RAM):
detector = FootballObjectDetector(model_size="nano")  # Smallest
video = cv2.resize(frame, (640, 480))  # Lower resolution
min_commentary_interval = 3.0  # Longer gaps


# ============================================================================
# 2. CLOUD DEPLOYMENT (AWS EC2 / Google Cloud)
# ============================================================================

"""
Deploy on cloud GPU for scalability
"""

# AWS DEPLOYMENT STEPS:
# ==============================

# 1. Launch EC2 instance
# - AMI: Deep Learning AMI (with PyTorch, CUDA)
# - Instance: g4dn.xlarge (GPU) or g4dn.2xlarge (better performance)
# - Storage: 50GB EBS
# - Security group: Allow port 8000 for streaming

# 2. SSH into instance
ssh -i "your-key.pem" ubuntu@ec2-instance-ip

# 3. Install dependencies
sudo apt update
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Install Ollama or set OpenAI key
export OPENAI_API_KEY="sk-your-key"
# OR for local:
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull mistral

# 5. Run as service
python full_system.py s3://bucket/match.mp4 openai

# 6. Stream output (for broadcast)
pip install flask-streaming
# See section "5. STREAMING SERVER" below


# ============================================================================
# 3. DOCKER DEPLOYMENT (Containerized)
# ============================================================================

"""
Package as Docker container for portability
"""

# Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    libsm6 libxext6 libxrender-dev

WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV OLLAMA_HOST=http://ollama:11434

# Run system
ENTRYPOINT ["python", "full_system.py"]
CMD ["football_match.mp4", "ollama"]
EOF

# docker-compose.yml (with Ollama service)
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: ollama serve

  commentary:
    build: .
    container_name: football_commentary
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./videos:/app/videos
      - ./commentary_logs:/app/logs
    command: python full_system.py videos/match.mp4 ollama

volumes:
  ollama_data:
EOF

# Build and run
docker-compose build
docker-compose up

# Push to registry
docker tag football-commentary:latest myregistry.azurecr.io/football-commentary:v1
docker push myregistry.azurecr.io/football-commentary:v1


# ============================================================================
# 4. EDGE DEPLOYMENT (Jetson / Apple Silicon)
# ============================================================================

"""
Deploy on edge devices for on-premise, low-latency solutions
Jetson Nano: $99, can run mini models
"""

# JETSON NANO SETUP:
# ==================

# 1. Flash JetPack OS (includes CUDA)
# Download from: https://developer.nvidia.com/jetson-nano-developer-kit

# 2. On JetPack, install dependencies
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/fzlooay8zs7g1x5swvd2d12qcc2xqcd3.whl -O torch-2.0.0+nv23.05-cp310-no-auditwheel_linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp310-no-auditwheel_linux_aarch64.whl

# 3. Install other requirements (skip OpenAI if offline)
pip install opencv-python ultralytics edge-tts pygame transformers

# 4. Use nano-sized models
detector = FootballObjectDetector(model_size="nano")

# 5. Run
python full_system.py 0 ollama  # 0 = camera input


# APPLE SILICON DEPLOYMENT:
# =========================

# Install with ARM64 support
pip install torch::2.0.0 torchvision::0.15.1 -c pytorch

# Use Metal Performance Shaders (faster than CPU)
import torch
torch.mps.is_available()

# Run with MPS
device = "mps" if torch.mps.is_available() else "cpu"


# ============================================================================
# 5. STREAMING SERVER (Broadcasting output)
# ============================================================================

"""
Stream commentary and video over network
For integration with broadcast systems
"""

from flask import Flask, Response, jsonify
import cv2
import threading
from full_system import FullCommentarySystem

app = Flask(__name__)
system = None
latest_frame = None
latest_commentary = None

def run_system_thread():
    """Run commentary system in background"""
    global system
    system = FullCommentarySystem(
        video_source="rtsp://camera-ip/stream",
        generator_backend="openai",
        enable_voice=False  # Disable audio in server
    )
    system.run()

@app.route('/video_feed')
def video_feed():
    """Stream video with analysis overlays"""
    def generate():
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/commentary')
def get_commentary():
    """Get latest commentary as JSON"""
    return jsonify({
        "commentary": latest_commentary,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    if system:
        return jsonify(system.get_statistics())
    return {"error": "System not running"}

if __name__ == '__main__':
    # Start system in background
    thread = threading.Thread(target=run_system_thread, daemon=True)
    thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=False)

# Client to consume stream:
# ========================
# HTML/JavaScript viewer:
cat > index.html << 'EOF'
<html>
  <body>
    <h1>Football Commentary</h1>
    <img src="/video_feed" width="800">
    <p id="commentary"></p>
    
    <script>
      fetch('/commentary')
        .then(r => r.json())
        .then(data => {
          document.getElementById('commentary').textContent = data.commentary;
        });
      
      setInterval(() => {
        fetch('/stats')
          .then(r => r.json())
          .then(data => console.log(data));
      }, 5000);
    </script>
  </body>
</html>
EOF

# Run streaming server
python stream_server.py

# Access from browser:
# http://localhost:8000


# ============================================================================
# 6. BROADCAST INTEGRATION (OBS / NDI)
# ============================================================================

"""
Integrate with streaming platforms and broadcast software
"""

# OBS (Open Broadcaster Software) Integration:
# =============================================

# Option 1: Capture desktop/window
# - Run full_system.py with visualization
# - In OBS, add "Window Capture" source
# - Select commentary system window

# Option 2: Network stream (better)
# - Run stream_server.py above
# - In OBS, add "Media Source"
# - URL: http://localhost:8000/video_feed

# Option 3: NDI (New Definition of Independence)
# - Requires: pip install python-ndisend
from pyndi import send
ndi_sender = send.send_create()

while True:
    frame = # Get from system
    ndi_sender.send(frame)  # Send to NDI network

# Then in OBS:
# - Install NDI plugin
# - Add NDI source -> select "Football Commentary"


# ============================================================================
# 7. DATABASE & ANALYTICS LOGGING
# ============================================================================

"""
Log all commentary + metrics for analysis
"""

import sqlite3
from datetime import datetime

def setup_analytics_db():
    """Create database for logging"""
    conn = sqlite3.connect('football_analytics.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS commentary
                 (id INTEGER PRIMARY KEY,
                  timestamp TEXT,
                  minute INTEGER,
                  player TEXT,
                  action TEXT,
                  commentary TEXT,
                  confidence REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS performance
                 (id INTEGER PRIMARY KEY,
                  timestamp TEXT,
                  fps REAL,
                  detection_time_ms REAL,
                  commentary_time_ms REAL)''')
    
    conn.commit()
    return conn

def log_commentary(conn, minute, player, action, commentary, confidence):
    """Log generated commentary"""
    c = conn.cursor()
    c.execute('''INSERT INTO commentary VALUES
                 (NULL, ?, ?, ?, ?, ?, ?)''',
              (datetime.now(), minute, player, action, commentary, confidence))
    conn.commit()

def generate_match_report(conn):
    """Generate post-match analysis report"""
    c = conn.cursor()
    
    # Top actions
    c.execute('''SELECT action, COUNT(*) as count
                 FROM commentary
                 GROUP BY action
                 ORDER BY count DESC''')
    print("Action Summary:", c.fetchall())
    
    # Player performance
    c.execute('''SELECT player, COUNT(*) as plays
                 FROM commentary
                 GROUP BY player
                 ORDER BY plays DESC
                 LIMIT 10''')
    print("Top Players:", c.fetchall())
    
    # Performance metrics
    c.execute('''SELECT AVG(fps), AVG(detection_time_ms)
                 FROM performance''')
    print("System Performance:", c.fetchone())


# ============================================================================
# 8. PRODUCTION CHECKLIST
# ============================================================================

"""
Before deploying to production:
"""

# Performance Testing
# - [ ] Test with 1+ hour of footage
# - [ ] Verify FPS >= 20
# - [ ] Check memory doesn't leak
# - [ ] Test with different video formats
# - [ ] Load test (multiple concurrent streams)

# Reliability
# - [ ] Error handling for corrupted frames
# - [ ] Fallback if commentary generation fails
# - [ ] Graceful shutdown on errors
# - [ ] Logging of all errors

# Security
# - [ ] API keys stored in environment variables
# - [ ] Validate video source URLs
# - [ ] Rate limiting on streaming endpoints
# - [ ] User authentication for sensitive endpoints

# Monitoring
# - [ ] Real-time metrics dashboard
# - [ ] Alert on low FPS / failed generation
# - [ ] Log all system events
# - [ ] Database backups

# Documentation
# - [ ] API documentation (if using server)
# - [ ] Setup guide for operators
# - [ ] Troubleshooting guide
# - [ ] Performance tuning guide


# ============================================================================
# 9. SCALING GUIDE
# ============================================================================

"""
For handling multiple concurrent streams
"""

from multiprocessing import Pool
from queue import Queue
import threading

def process_stream(stream_config):
    """Process single stream"""
    system = FullCommentarySystem(
        video_source=stream_config['video'],
        generator_backend=stream_config['backend']
    )
    system.run()
    return system.get_statistics()

def scale_to_multiple_streams(streams_config):
    """Process multiple streams in parallel"""
    with Pool(processes=4) as pool:  # 4 concurrent streams
        results = pool.map(process_stream, streams_config)
    return results

# Load balancing setup:
# Use nginx to distribute requests
nginx_config = """
upstream commentary_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://commentary_backend;
    }
}
"""


# ============================================================================
# 10. TROUBLESHOOTING DEPLOYMENT
# ============================================================================

# Issue: "Connection refused" on cloud
# Solution: Check security group allows port 8000
# aws ec2 authorize-security-group-ingress --group-id sg-xxxxxx --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Issue: "CUDA out of memory" on GPU
# Solution: Reduce batch size or model size
detector = FootballObjectDetector(model_size="nano")

# Issue: "Slow commentary generation"
# Solution: Use faster backend
backend = "ollama"  # Faster than OpenAI (no network latency)

# Issue: "Video drops frames"
# Solution: Reduce resolution or increase frame queue
frame_queue = queue.Queue(maxsize=50)  # Increase buffer

# Monitor resources:
# Linux: watch -n 1 nvidia-smi  # GPU
# Windows: tasklist | findstr python  # Check memory

print(f"✓ Deployment guide complete!")
