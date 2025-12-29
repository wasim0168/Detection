import os
import re
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

# Configuration with environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
USE_CUDA = os.environ.get("USE_CUDA", "false").lower() in ("1", "true", "yes")
DEVICE = "cuda" if USE_CUDA else "cpu"
INFERENCE_SIZE = int(os.environ.get("INFERENCE_SIZE", "320"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "1920"))
PORT = int(os.environ.get("PORT", "5000"))
HOST = os.environ.get("HOST", "0.0.0.0")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RT-Detect")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model: Optional[YOLO] = None
model_loaded = False
detection_stats = {
    "total_detections": 0,
    "total_processing_time": 0,
    "total_frames_processed": 0
}

class DetectionError(Exception):
    """Custom exception for detection errors"""
    pass

def setup_model() -> None:
    """Initialize and warm up the YOLO model"""
    global model, model_loaded
    
    try:
        logger.info(f"Loading model: {MODEL_NAME} on device: {DEVICE}")
        
        # Load model with optimized settings
        model = YOLO(MODEL_NAME)
        
        # Fuse conv and bn for faster inference
        model.fuse()
        
        # Model information
        logger.info(f"Model loaded successfully: {MODEL_NAME}")
        logger.info(f"Model classes: {len(model.names)}")
        logger.info(f"Model device: {DEVICE}")
        
        # Warm up the model
        logger.info("Warming up model...")
        warmup_iterations = 5 if USE_CUDA else 3
        
        for i in range(warmup_iterations):
            dummy_input = np.random.randint(0, 255, (INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
            results = model.predict(
                dummy_input,
                imgsz=INFERENCE_SIZE,
                device=DEVICE,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                augment=False
            )
            logger.info(f"Warmup iteration {i+1}/{warmup_iterations} completed")
        
        model_loaded = True
        logger.info("Model warmup complete and ready for inference!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False
        raise DetectionError(f"Model loading failed: {str(e)}")

def decode_base64_image(b64str: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array
    
    Args:
        b64str: Base64 encoded image string
        
    Returns:
        numpy.ndarray: Decoded image in RGB format
        
    Raises:
        DetectionError: If image decoding fails
    """
    try:
        # Handle data URI format
        data_uri_pattern = r"data:image/.+;base64,(.+)"
        match = re.match(data_uri_pattern, b64str)
        if match:
            b64str = match.group(1)
        
        # Decode base64
        image_data = base64.b64decode(b64str)
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        
        # Decode image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise DetectionError("Failed to decode image data")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    except Exception as e:
        raise DetectionError(f"Image decoding failed: {str(e)}")

def validate_image_size(img: np.ndarray) -> None:
    """
    Validate image dimensions
    
    Args:
        img: Input image as numpy array
        
    Raises:
        DetectionError: If image dimensions are invalid
    """
    height, width = img.shape[:2]
    
    if height == 0 or width == 0:
        raise DetectionError("Image has zero dimensions")
    
    if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        raise DetectionError(f"Image too large. Max allowed: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")

def preprocess_image(img: np.ndarray, target_size: int = 320) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    Preprocess image for YOLO inference
    
    Args:
        img: Input image in RGB format
        target_size: Target size for resizing
        
    Returns:
        tuple: (processed_image, scale_factor, original_size, padding)
    """
    original_height, original_width = img.shape[:2]
    original_size = (original_width, original_height)
    
    # Calculate scaling factor while maintaining aspect ratio
    scale = min(target_size / original_height, target_size / original_width)
    new_height, new_width = int(original_height * scale), int(original_width * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size (center padding)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=(114, 114, 114)  # YOLO training background color
    )
    
    return padded, scale, original_size, (left, top)

def postprocess_detections(
    detections: List[Dict[str, Any]], 
    scale: float, 
    padding: Tuple[int, int], 
    original_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """
    Convert detections back to original image coordinates
    
    Args:
        detections: List of detection results
        scale: Scaling factor used in preprocessing
        padding: Padding applied (left, top)
        original_size: Original image size (width, height)
        
    Returns:
        list: Processed detections in original coordinates
    """
    left_pad, top_pad = padding
    orig_width, orig_height = original_size
    
    processed_detections = []
    
    for det in detections:
        x_center, y_center, width, height = det['box']
        
        # Remove padding
        x_center = max(0, x_center - left_pad)
        y_center = max(0, y_center - top_pad)
        
        # Scale back to original image size
        x_center /= scale
        y_center /= scale
        width /= scale
        height /= scale
        
        # Convert to top-left coordinates
        x1 = max(0, x_center - width / 2)
        y1 = max(0, y_center - height / 2)
        x2 = min(orig_width, x_center + width / 2)
        y2 = min(orig_height, y_center + height / 2)
        
        # Ensure valid bounding box
        if x2 <= x1 or y2 <= y1:
            continue
            
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        processed_detections.append({
            'name': det['name'],
            'class': det.get('class', -1),
            'confidence': float(det['confidence']),
            'box': [float(x_center), float(y_center), float(width), float(height)],
            'bbox_pixels': [float(x1), float(y1), float(x2), float(y2)]
        })
    
    return processed_detections

@app.route("/detect", methods=["POST"])
def detect():
    """
    Main detection endpoint
    """
    start_time = time.time()
    
    if not model_loaded:
        return jsonify({"error": "Model not loaded", "status": "error"}), 503
    
    try:
        # Parse request data
        data = request.get_json(force=True)
        
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided", "status": "error"}), 400
        
        # Extract parameters with defaults
        confidence_threshold = float(data.get("confidence", CONFIDENCE_THRESHOLD))
        inference_size = int(data.get("imgsz", INFERENCE_SIZE))
        
        # Validate parameters
        if not 0.1 <= confidence_threshold <= 0.9:
            return jsonify({"error": "Confidence threshold must be between 0.1 and 0.9", "status": "error"}), 400
        
        if inference_size < 160 or inference_size > 1280:
            return jsonify({"error": "Image size must be between 160 and 1280", "status": "error"}), 400
        
        # Decode image
        decode_start = time.time()
        img = decode_base64_image(data["image"])
        decode_time = time.time() - decode_start
        
        # Validate image
        validate_image_size(img)
        
        # Preprocess
        preprocess_start = time.time()
        img_processed, scale, orig_size, padding = preprocess_image(img, inference_size)
        preprocess_time = time.time() - preprocess_start
        
        # Inference
        inference_start = time.time()
        results = model.predict(
            img_processed, 
            imgsz=inference_size, 
            device=DEVICE, 
            conf=confidence_threshold, 
            verbose=False,
            augment=False,
            max_det=50  # Limit maximum detections
        )
        inference_time = time.time() - inference_start
        
        # Process results
        process_start = time.time()
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if box.conf[0] >= confidence_threshold:
                        xywh = box.xywh[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = model.names.get(cls, str(cls))
                        
                        detections.append({
                            "name": name,
                            "class": cls,
                            "confidence": conf,
                            "box": xywh
                        })
        
        # Convert coordinates back to original image space
        final_detections = postprocess_detections(detections, scale, padding, orig_size)
        process_time = time.time() - process_start
        
        # Update statistics
        detection_stats["total_detections"] += len(final_detections)
        detection_stats["total_frames_processed"] += 1
        
        total_time = time.time() - start_time
        detection_stats["total_processing_time"] += total_time
        
        # Prepare response
        response = {
            "status": "success",
            "detections": final_detections, 
            "width": orig_size[0], 
            "height": orig_size[1],
            "count": len(final_detections),
            "timing": {
                "total_ms": round(total_time * 1000, 2),
                "decode_ms": round(decode_time * 1000, 2),
                "preprocess_ms": round(preprocess_time * 1000, 2),
                "inference_ms": round(inference_time * 1000, 2),
                "process_ms": round(process_time * 1000, 2)
            },
            "model_info": {
                "name": MODEL_NAME,
                "device": DEVICE,
                "inference_size": inference_size,
                "confidence_threshold": confidence_threshold
            }
        }
        
        logger.info(f"Detection completed: {len(final_detections)} objects in {response['timing']['total_ms']}ms")
        
        return jsonify(response)

    except DetectionError as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}", "status": "error"}), 500

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    """
    health_status = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "timestamp": time.time(),
        "stats": {
            "total_frames_processed": detection_stats["total_frames_processed"],
            "total_detections": detection_stats["total_detections"],
            "average_processing_time": round(
                detection_stats["total_processing_time"] / max(detection_stats["total_frames_processed"], 1), 
                4
            )
        }
    }
    
    return jsonify(health_status)

@app.route("/info", methods=["GET"])
def info():
    """
    Model and system information endpoint
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    
    info = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "inference_size": INFERENCE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_image_size": MAX_IMAGE_SIZE,
        "model_classes": len(model.names) if model else 0,
        "class_names": model.names if model else {}
    }
    
    return jsonify(info)

@app.route("/stats", methods=["GET"])
def stats():
    """
    Detection statistics endpoint
    """
    stats_data = {
        "detection_stats": detection_stats,
        "performance_metrics": {
            "average_detections_per_frame": round(
                detection_stats["total_detections"] / max(detection_stats["total_frames_processed"], 1), 
                2
            ),
            "average_processing_time_ms": round(
                (detection_stats["total_processing_time"] / max(detection_stats["total_frames_processed"], 1)) * 1000, 
                2
            )
        }
    }
    
    return jsonify(stats_data)

@app.route("/reload", methods=["POST"])
def reload_model():
    """
    Reload model endpoint (for development)
    """
    try:
        global model, model_loaded
        logger.info("Reloading model...")
        
        # Clean up existing model
        if model:
            del model
        
        # Reload model
        setup_model()
        
        return jsonify({"status": "success", "message": "Model reloaded successfully"})
        
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        return jsonify({"error": f"Model reload failed: {str(e)}", "status": "error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed", "status": "error"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

def main():
    """
    Main application entry point
    """
    try:
        # Initialize model
        setup_model()
        
        # Print startup information
        logger.info("=" * 50)
        logger.info("RT-Detect Server Starting Up")
        logger.info("=" * 50)
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Inference Size: {INFERENCE_SIZE}")
        logger.info(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Server: {HOST}:{PORT}")
        logger.info("=" * 50)
        
        # Start Flask server
        app.run(
            host=HOST, 
            port=PORT, 
            debug=False, 
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested...")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
    finally:
        logger.info("RT-Detect Server shutting down...")

if __name__ == "__main__":
    main()