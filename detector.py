import os, re, base64, cv2, numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import time
import threading

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
USE_CUDA = os.environ.get("USE_CUDA", "false").lower() in ("1", "true")
DEVICE = "cuda" if USE_CUDA else "cpu"
INFERENCE_SIZE = int(os.environ.get("INFERENCE_SIZE", "320"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))


app = Flask(__name__)

# Load model with optimized settings
model = YOLO(MODEL_NAME)
model.fuse()  # Fuse conv and bn for faster inference

# Correct warmup - single image format
print(f"Warming up model on {DEVICE}...")
dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
for _ in range(3):  # Multiple warmup runs
    model.predict(dummy, imgsz=INFERENCE_SIZE, device=DEVICE, 
                 verbose=False, conf=CONFIDENCE_THRESHOLD)

print("Warmup complete!")

DATA_URI_RE = re.compile(r"data:image/.+;base64,(.+)")

def decode_base64_image(b64str):
    """Base64 image decoding"""
    m = DATA_URI_RE.match(b64str)
    if m: 
        b64str = m.group(1)
    
    b = base64.b64decode(b64str)
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(img, target_size=320):
    """Image preprocessing with aspect ratio preservation"""
    h, w = img.shape[:2]
    
    # Calculate scaling factor while maintaining aspect ratio
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size (center padding)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return padded, scale, (w, h), (left, top)

def postprocess_detections(detections, scale, padding, orig_size):
    """Convert detections back to original image coordinates"""
    left_pad, top_pad = padding
    orig_w, orig_h = orig_size
    
    processed = []
    for det in detections:
        # Remove padding and scale back to original coordinates
        x_center, y_center, width, height = det['box']
        
        # Remove padding
        x_center = max(0, x_center - left_pad)
        y_center = max(0, y_center - top_pad)
        
        # Scale back to original image size
        x_center /= scale
        y_center /= scale
        width /= scale
        height /= scale
        
        # Ensure coordinates are within image bounds
        x_center = min(max(x_center, 0), orig_w)
        y_center = min(max(y_center, 0), orig_h)
        width = min(max(width, 0), orig_w)
        height = min(max(height, 0), orig_h)
        
        processed.append({
            'name': det['name'],
            'confidence': det['confidence'],
            'box': [x_center, y_center, width, height]
        })
    
    return processed

@app.route("/detect", methods=["POST"])
def detect():
    start_time = time.time()
    data = request.get_json(force=True)
    
    if "image" not in data:
        return jsonify({"error": "No image"}), 400

    try:
        # Decode image
        decode_start = time.time()
        img = decode_base64_image(data["image"])
        decode_time = time.time() - decode_start

        # Preprocess
        preprocess_start = time.time()
        img_processed, scale, orig_size, padding = preprocess_image(img, INFERENCE_SIZE)
        preprocess_time = time.time() - preprocess_start

        # Inference
        inference_start = time.time()
        results = model.predict(
            img_processed, 
            imgsz=INFERENCE_SIZE, 
            device=DEVICE, 
            conf=CONFIDENCE_THRESHOLD, 
            verbose=False,
            augment=False
        )
        inference_time = time.time() - inference_start

        # Process results
        process_start = time.time()
        detections = []
        w0, h0 = orig_size
        
        for r in results:
            for box in r.boxes:
                if box.conf[0] >= CONFIDENCE_THRESHOLD:
                    xywh = box.xywh[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = model.names.get(cls, str(cls))

                    detections.append({
                        "name": name,
                        "confidence": conf,
                        "box": xywh
                    })
        
        # Convert coordinates back to original image space
        final_detections = postprocess_detections(detections, scale, padding, orig_size)
        process_time = time.time() - process_start

        total_time = time.time() - start_time
        
        response = {
            "detections": final_detections, 
            "width": w0, 
            "height": h0,
            "timing": {
                "total_ms": round(total_time * 1000, 2),
                "decode_ms": round(decode_time * 1000, 2),
                "preprocess_ms": round(preprocess_time * 1000, 2),
                "inference_ms": round(inference_time * 1000, 2),
                "process_ms": round(process_time * 1000, 2)
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": DEVICE})

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": MODEL_NAME,
        "device": DEVICE,
        "inference_size": INFERENCE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

if __name__ == "__main__":
    print(f"Starting server on 0.0.0.0:5000")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Inference size: {INFERENCE_SIZE}")


    
    # Use development server for testing
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)