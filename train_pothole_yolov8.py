# ============================================================
# Pothole Detection with YOLOv8 (PyTorch)
# Run this in Google Colab: Runtime > Change runtime type > GPU (T4)
# Paste each section into a separate Colab cell, or run as-is with `!python`
# ============================================================

# --- Cell 1: Install dependencies ---
# !pip install ultralytics roboflow -q

# --- Cell 2: Download dataset from Roboflow Universe ---
# Sign up free at https://roboflow.com, get your API key from Settings > API Keys
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("gerapothole").project("pothole-detection-yolov8")
dataset = project.version(1).download("yolov8")
# This gives you dataset.location -> path to data.yaml, train/, valid/, test/

# --- Cell 3: Train YOLOv8 on the dataset ---
from ultralytics import YOLO

# Start from a pretrained checkpoint (transfer learning) - fast + effective on small datasets
model = YOLO("yolov8n.pt")  # nano version: fast to train, good for a portfolio project

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="pothole-detector",
    patience=10,       # early stopping if no improvement
)

# --- Cell 4: Evaluate the model ---
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")

# --- Cell 5: Run inference on a test image and save the result ---
test_results = model.predict(
    source=f"{dataset.location}/test/images",
    save=True,
    conf=0.4,
)
# Saved outputs will be in runs/detect/predict/ - grab a few sample images for your README

# --- Cell 6: Export the trained weights ---
# Best weights are auto-saved at: runs/detect/pothole-detector/weights/best.pt
# Download this file and include it in your GitHub repo (or link via Git LFS / releases if large)

# --- Cell 7 (optional, if you have time): Run on a short video/webcam-style clip ---
# model.predict(source="path/to/test_video.mp4", save=True)
# This is a nice bonus artifact for the README - shows real-time-style detection

print("Done. Copy best.pt, a few prediction images, and results.png (training curves) into your repo.")
