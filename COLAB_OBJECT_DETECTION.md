# Google Colab: YOLOv8 Football Detection Training

This guide trains a custom YOLOv8 model to detect **players, ball, goalkeeper, and referee** in football match footage. Once trained, you'll download the `.pt` weights file and plug it into your `object_detector.py`.

---

## **Cell 1: Install Dependencies** (~1 min)

```python
!pip install -q ultralytics roboflow
import torch
print(f"✓ Installed! GPU available: {torch.cuda.is_available()}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## **Cell 2: Download Football Dataset from Roboflow**

> [!NOTE]
> This uses the **football-players-detection** dataset from Roboflow Universe.
> It contains labeled images with bounding boxes for: `player`, `ball`, `goalkeeper`, `referee`.
> You need a free Roboflow account — get your API key at https://app.roboflow.com/settings/api

```python
from roboflow import Roboflow

# Replace with YOUR Roboflow API key (free tier works fine)
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"  # <-- CHANGE THIS

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Download the football players detection dataset in YOLOv8 format
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov8")

print(f"✓ Dataset downloaded!")
print(f"  Location: {dataset.location}")
```

> [!TIP]
> If the above dataset doesn't work or you want a different one, search for
> "football player ball detection" on https://universe.roboflow.com and swap
> the workspace/project/version IDs above.

---

## **Cell 3: Train YOLOv8** (~15-30 min on Colab GPU)

```python
from ultralytics import YOLO

# Load YOLOv8-small pretrained weights (transfer learning)
model = YOLO("yolov8s.pt")

# Train on our football dataset
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,               # 50 epochs is a good starting point
    imgsz=640,               # Standard YOLO input size
    batch=16,                # Adjust down to 8 if you get OOM errors
    name="football_detector",
    patience=10,             # Stop early if no improvement for 10 epochs
    device=0,                # Use GPU
    workers=2,               # Colab has limited CPU workers
    plots=True,              # Generate training plots
)

print("✓ Training complete!")
```

---

## **Cell 4: Evaluate Model Quality**

```python
# Run validation on the test set
metrics = model.val()

print(f"\n📊 Results:")
print(f"  mAP50:     {metrics.box.map50:.3f}")
print(f"  mAP50-95:  {metrics.box.map:.3f}")
print(f"  Precision:  {metrics.box.mp:.3f}")
print(f"  Recall:     {metrics.box.mr:.3f}")
```

---

## **Cell 5: Test on Sample Images**

```python
import glob
from IPython.display import Image, display

# Run inference on test images
test_images = glob.glob(f"{dataset.location}/test/images/*.jpg")[:5]

if test_images:
    results = model.predict(
        source=test_images,
        save=True,
        conf=0.4,
        project="test_results"
    )
    
    # Display results
    for img_path in glob.glob("test_results/predict/*.jpg")[:5]:
        display(Image(filename=img_path, width=600))
        print("---")
else:
    print("No test images found. Try with a sample football image URL:")
    results = model.predict(
        source="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Soccer_field_-_empty.jpg/1280px-Soccer_field_-_empty.jpg",
        save=True, conf=0.4, project="test_results"
    )
```

---

## **Cell 6: Export Trained Weights**
*Download the `.pt` file to use in your local `object_detector.py`.*

```python
from google.colab import files
import shutil
import os

# The best weights are saved here
best_weights = "runs/detect/football_detector/weights/best.pt"

if os.path.exists(best_weights):
    # Copy to a clean name
    shutil.copy(best_weights, "football_yolov8s_best.pt")
    
    print("📥 Downloading trained weights...")
    files.download("football_yolov8s_best.pt")
    print("✓ Download started! Check your browser downloads.")
else:
    print("⚠️ Weights not found. Check training output above for errors.")
    # List available weight files
    for root, dirs, fls in os.walk("runs/detect"):
        for f in fls:
            if f.endswith(".pt"):
                print(f"  Found: {os.path.join(root, f)}")
```

---

## **Next Steps**

Once downloaded:
1. Place `football_yolov8s_best.pt` in your `football-commentary/` project folder
2. Update `object_detector.py` to load your custom weights instead of the generic COCO model
3. Train the **Event Classifier** (Tier 2) to recognize actions from detection sequences
