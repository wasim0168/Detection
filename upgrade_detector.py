# simple_detector.py - Lightweight version with simulated detection
import os
import re
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
import random

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simple-Detect")

# Weapons keywords for detection
WEAPON_KEYWORDS = [
    'knife', 'gun', 'pistol', 'rifle', 'shotgun', 'sword', 'axe', 
    'bat', 'hammer', 'crowbar', 'bomb', 'grenade', 'blade', 'dagger',
    'machete', 'revolver', 'handgun', 'weapon', 'harmful'
]

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(force=True)
        
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Simulate detection processing
        time.sleep(0.05)  # Simulate processing time
        
        # Generate random detections for demonstration
        detection_types = ['person', 'car', 'knife', 'gun', 'bicycle', 'cell phone', 'backpack']
        
        # Randomly decide if weapons are present
        has_weapons = random.random() > 0.7
        
        detections = []
        weapons_detected = []
        
        # Generate 2-5 random detections
        num_detections = random.randint(2, 6)
        
        for i in range(num_detections):
            name = random.choice(detection_types)
            confidence = random.uniform(0.4, 0.95)
            
            # Check if it's a weapon
            is_weapon = any(keyword in name.lower() for keyword in WEAPON_KEYWORDS)
            
            detection = {
                "name": name,
                "class": i,
                "confidence": confidence,
                "box": [
                    random.randint(100, 500),  # x_center
                    random.randint(100, 300),  # y_center
                    random.randint(50, 150),   # width
                    random.randint(50, 200)    # height
                ],
                "is_weapon": is_weapon,
                "weapon_type": name if is_weapon else None,
                "threat_level": "critical" if is_weapon and name in ['gun', 'bomb'] else "high" if is_weapon else None
            }
            
            detections.append(detection)
            
            if is_weapon:
                weapons_detected.append(detection)
        
        # Force weapons detection occasionally for demo
        if has_weapons and len(weapons_detected) == 0:
            weapon_detection = {
                "name": random.choice(['knife', 'gun']),
                "class": 99,
                "confidence": random.uniform(0.7, 0.95),
                "box": [300, 200, 80, 120],
                "is_weapon": True,
                "weapon_type": random.choice(['knife', 'gun']),
                "threat_level": "critical" if random.random() > 0.5 else "high"
            }
            detections.append(weapon_detection)
            weapons_detected.append(weapon_detection)
        
        # Calculate threat summary
        threat_summary = {}
        for weapon in weapons_detected:
            level = weapon.get('threat_level', 'low')
            threat_summary[level] = threat_summary.get(level, 0) + 1
        
        response = {
            "status": "success",
            "detections": detections,
            "weapons_detected": weapons_detected,
            "weapons_count": len(weapons_detected),
            "threat_summary": threat_summary,
            "width": 640,
            "height": 480,
            "count": len(detections),
            "timing": {
                "total_ms": random.randint(50, 150),
                "decode_ms": random.randint(5, 20),
                "preprocess_ms": random.randint(5, 15),
                "inference_ms": random.randint(30, 80),
                "process_ms": random.randint(5, 20)
            },
            "model_info": {
                "name": "simulation_mode",
                "device": "cpu",
                "inference_size": 320,
                "confidence_threshold": 0.5,
                "weapons_detection_enabled": True
            }
        }
        
        if weapons_detected:
            logger.warning(f"⚠️ WEAPONS DETECTED: {[w['name'] for w in weapons_detected]}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "weapons_model_loaded": True,
        "device": "cpu",
        "timestamp": time.time()
    })

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": "simulation_mode",
        "weapons_model": "simulation_mode",
        "device": "cpu",
        "weapons_categories": {
            "knife": ["knife", "dagger", "blade"],
            "gun": ["gun", "pistol", "rifle"]
        },
        "threat_levels": {
            "gun": "critical",
            "knife": "high"
        }
    })

if __name__ == "__main__":
    print("=" * 50)
    print("Simple RT-Detect Server with Weapons Detection")
    print("=" * 50)
    print("Server running on http://localhost:5000")
    print("Weapons detection: ENABLED (simulation mode)")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    """
    health_status = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "weapons_model_loaded": weapons_model_loaded,
        "device": DEVICE,
        "timestamp": time.time(),
        "stats": {
            "total_frames_processed": detection_stats["total_frames_processed"],
            "total_detections": detection_stats["total_detections"],
            "weapons_detected": detection_stats["weapons_detected"],
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
        "weapons_model": WEAPONS_MODEL_NAME if weapons_model_loaded else "Not loaded",
        "device": DEVICE,
        "inference_size": INFERENCE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "weapons_confidence_threshold": WEAPONS_CONFIDENCE_THRESHOLD,
        "max_image_size": MAX_IMAGE_SIZE,
        "model_classes": len(model.names) if model else 0,
        "class_names": model.names if model else {},
        "weapons_categories": WEAPONS_CATEGORIES,
        "threat_levels": THREAT_LEVELS
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
            ),
            "weapons_detection_rate": round(
                detection_stats["weapons_detected"] / max(detection_stats["total_frames_processed"], 1) * 100,
                2
            )
        }
    }
    
    return jsonify(stats_data)

@app.route("/weapons/config", methods=["GET", "POST"])
def weapons_config():
    global WEAPONS_CONFIDENCE_THRESHOLD

    if request.method == "GET":
        return jsonify({
            "weapons_categories": WEAPONS_CATEGORIES,
            "threat_levels": THREAT_LEVELS,
            "confidence_threshold": WEAPONS_CONFIDENCE_THRESHOLD,
            "model_loaded": weapons_model_loaded
        })
    
    elif request.method == "POST":
        try:
            data = request.get_json()

            if "confidence_threshold" in data:
                WEAPONS_CONFIDENCE_THRESHOLD = float(data["confidence_threshold"])
                logger.info(
                    f"Weapons confidence threshold updated to {WEAPONS_CONFIDENCE_THRESHOLD}"
                )
            
            return jsonify({
                "status": "success",
                "weapons_categories": WEAPONS_CATEGORIES,
                "threat_levels": THREAT_LEVELS,
                "confidence_threshold": WEAPONS_CONFIDENCE_THRESHOLD
            })

        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 400

@app.route("/reload", methods=["POST"])
def reload_model():
    """
    Reload model endpoint (for development)
    """
    try:
        global model, model_loaded, weapons_model, weapons_model_loaded
        logger.info("Reloading models...")
        
        # Clean up existing models
        if model:
            del model
        if weapons_model:
            del weapons_model
        
        # Reload models
        setup_model()
        setup_weapons_model()
        
        return jsonify({"status": "success", "message": "Models reloaded successfully"})
        
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
        # Initialize models
        setup_model()
        setup_weapons_model()
        
        # Print startup information
        logger.info("=" * 50)
        logger.info("RT-Detect Server Starting Up")
        logger.info("=" * 50)
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Weapons Model: {WEAPONS_MODEL_NAME if weapons_model_loaded else 'Disabled'}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Inference Size: {INFERENCE_SIZE}")
        logger.info(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Weapons Confidence: {WEAPONS_CONFIDENCE_THRESHOLD}")
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