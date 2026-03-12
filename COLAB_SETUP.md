# Google Colab Setup Guide

## Step 1: Prepare Files Locally

Run this cleanup script to remove unnecessary files:

```powershell
cd c:\Users\Murta\OneDrive\Desktop\football-commentary

# Remove cloud/local deployment files
$filesToRemove = @(
    'aws_manage.ps1',
    'setup_aws_instance.sh',
    'modal_app.py',
    'stream_server.py',
    'quickstart.py',
    'test_tts.py',
    'test_output.mp3',
    'commentary.wav',
    'CLOUD_DEPLOYMENT_SIMPLE.md',
    'CLOUD_QUICK_START.md',
    'CLOUD_SUMMARY.md',
    'API_EXAMPLES.py',
    'QUICKSTART_REFERENCE.md'
)

foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force
    }
}

# Remove venv
if (Test-Path 'venv') {
    Remove-Item -Recurse -Force 'venv'
}

Write-Host "✓ Cleanup complete!"
```

## Step 2: Upload to Google Colab

You'll have these files remaining:
```
📁 .git/
📁 .vscode/
📄 full_system.py              ← Main integrated system
📄 object_detector.py          ← YOLO detection
📄 commentary_generator.py     ← LLM commentary
📄 commentator.py              ← TTS voice synthesis
📄 real_time_pipeline.py       ← Multi-threaded pipeline
📄 tests.py                    ← Unit tests
📄 requirements.txt            ← Dependencies
📄 README.md
📄 ARCHITECTURE.md
📄 DEPLOYMENT_GUIDE.md
📄 TRAINING_GUIDE.md
```

**Upload method:**
1. Go to Google Drive
2. Create new folder: `football-commentary`
3. Upload these files

OR use Git:
```bash
git push origin main
# Then in Colab: !git clone <your-repo-url>
```

## Step 3: Google Colab Notebooks

### Notebook 1: Testing with Pre-Trained Models

```python
# Cell 1: Install dependencies (takes 2-3 min)
!pip install torch torchvision ultralytics opencv-python edge-tts pygame transformers requests python-dotenv albumentations -q

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create symlink to your uploaded files
import os
os.chdir('/content/drive/My Drive/football-commentary')

# Cell 3: Import modules
import sys
sys.path.insert(0, '/content/drive/My Drive/football-commentary')

from object_detector import FootballObjectDetector
from commentary_generator import ContextAwareCommentator
from commentator import Commentator
from full_system import FullCommentarySystem

print("✓ Imports successful!")

# Cell 4: Test object detection
detector = FootballObjectDetector(model_size='m')
print(f"✓ YOLO model loaded: {detector.model}")

# Cell 5: Test commentary generation
commentator = ContextAwareCommentator(backend='huggingface')
print("✓ Commentary generator ready")

# Cell 6: Download a test video
!wget -q https://path-to-your-test-video.mp4 -O test_video.mp4
print("✓ Test video ready")

# Cell 7: Run full system
system = FullCommentarySystem(
    model_size='m',
    commentator_backend='huggingface',
    use_gpu=True
)

print("Processing video...")
stats = system.process_video('test_video.mp4', output_audio='output.wav')

print(f"""
✓ Processing complete!
  Frames processed: {stats['frames_processed']}
  Commentaries generated: {stats['commentaries_generated']}
  Average FPS: {stats['avg_fps']:.2f}
  Total time: {stats['total_time']:.2f}s
""")

# Cell 8: Download output
from google.colab import files
files.download('output.wav')
```

### Notebook 2: Fine-Tuning YOLO (Optional)

```python
# Cell 1: Setup
!pip install -q roboflow

# Cell 2: Get football dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("football-detection").project("football-players")
dataset = project.download("yolov8")

# Cell 3: Fine-tune YOLO
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # Load pre-trained

results = model.train(
    data='path/to/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,  # GPU
    patience=20,
    save=True
)

# Cell 4: Test fine-tuned model
best_model = YOLO('runs/detect/train/weights/best.pt')
results = best_model.predict('test_video.mp4', conf=0.5)

# Cell 5: Save model
best_model.export(format='onnx')
!wget -q https://drive.google.com/uc?export_format=zip&id=... -O best_model.zip
files.download('best_model.zip')
```

### Notebook 3: Fine-Tuning Commentary Generator (Optional)

```python
# Cell 1: Setup
!pip install -q datasets transformers

# Cell 2: Load your training data
from datasets import load_dataset

# Prepare your dataset in this format:
# {
#   "game_state": "2-1, 45th minute",
#   "action": "Player shoots at goal",
#   "commentary": "AND HE SHOOTS! WHAT A STRIKE! IT'S A GOAL!"
# }

dataset = load_dataset('csv', data_files='commentary_data.csv')

# Cell 3: Fine-tune model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

training_args = TrainingArguments(
    output_dir='./commentary_model',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

trainer.train()

# Cell 4: Save model
model.save_pretrained('commentary_model_fine_tuned')
!zip -r commentary_model.zip commentary_model_fine_tuned
files.download('commentary_model.zip')
```

## Step 4: Full End-to-End Workflow

```python
# Complete Colab workflow combining everything

# Input: Video file
# Output: Commented video with audio and stats

import cv2
import numpy as np
from full_system import FullCommentarySystem

# Initialize system with best models
system = FullCommentarySystem(
    model_size='m',
    commentator_backend='huggingface',
    use_gpu=True
)

# Process video
video_path = 'test_video.mp4'
output_audio = 'commentary.wav'

print("🎬 Processing video...")
stats = system.process_video(video_path, output_audio)

print(f"""
✅ Complete!
   Frames: {stats['frames_processed']}
   Commentaries: {stats['commentaries_generated']}
   FPS: {stats['avg_fps']:.1f}
   Time: {stats['total_time']:.1f}s
   Audio: {output_audio}
""")

# Download results
from google.colab import files
files.download(output_audio)
files.download('overlay_video.mp4')  # If you save with overlay
```

## Step 5: Performance Tips for Colab

### Use Nano Models (Faster)
```python
system = FullCommentarySystem(
    model_size='n',  # nano - 50% faster, less accurate
    use_gpu=True
)
```

### Enable Mixed Precision
```python
import torch
torch.cuda.set_bf16_enabled(True)  # Use BF16 for speed
```

### Process in Chunks
```python
# Split long videos into segments
segment_length = 300  # 5 min chunks
results = []
for start in range(0, total_frames, segment_length * fps):
    chunk = video[start:start+segment_length*fps]
    result = system.process_video(chunk)
    results.append(result)
```

## Troubleshooting

**"Out of memory"**
→ Use model_size='n' (nano) instead of 'm' (medium)

**"CUDA out of memory"**
→ Reduce batch size or use smaller model

**"Module not found"**
→ Make sure all .py files are in same directory as notebook

**"Ollama connection refused"**
→ Use 'huggingface' backend instead in Colab (Ollama requires server setup)

## Next Steps

1. **Clean up local files** ← Run cleanup script
2. **Upload to Google Drive** ← Upload remaining files
3. **Test with pre-trained models** ← Run Notebook 1
4. **Fine-tune if needed** ← Run Notebook 2 or 3
5. **Process your matches** ← Use full system

All done! You now have a Colab-ready football commentary system.
