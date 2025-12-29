import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import time
import logging
from scipy.spatial import distance as dist
import base64
import io
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Drowsiness-Detector")

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmarks for EAR calculation (MediaPipe indices)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Complete eye contours for visualization
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

class DrowsinessDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.ear_threshold = 0.25
        self.eye_state_history = []
        self.history_size = 5

    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) - More accurate formula"""
        try:
            # Vertical distances (more points for better accuracy)
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            
            # Horizontal distance
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            
            # EAR formula
            ear = (A + B) / (2.0 * C)
            return max(0, min(ear, 0.5))  # Clamp between 0 and 0.5
        except Exception as e:
            logger.warning(f"EAR calculation error: {e}")
            return 0.3  # Default open eye value

    def get_eye_landmarks(self, face_landmarks, image_shape, indices):
        """Extract specific eye landmarks"""
        points = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            points.append((x, y))
        return points

    def get_eye_contour(self, face_landmarks, image_shape, indices):
        """Extract complete eye contour for visualization"""
        points = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            points.append((x, y))
        return points

    def smooth_eye_state(self, current_state):
        """Apply smoothing to prevent flickering"""
        self.eye_state_history.append(current_state)
        if len(self.eye_state_history) > self.history_size:
            self.eye_state_history.pop(0)
        
        # Return majority state from history
        if len(self.eye_state_history) >= 3:
            open_count = sum(1 for state in self.eye_state_history if state == "open")
            return "open" if open_count > len(self.eye_state_history) // 2 else "closed"
        return current_state

    def detect_eyes(self, image_rgb):
        """Main eye detection function"""
        results = self.face_mesh.process(image_rgb)
        
        detection_data = {
            "faces_detected": 0,
            "eyes_detected": False,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "ear_avg": 0.0,
            "eye_state": "unknown",
            "is_drowsy": False,
            "confidence": 0.0,
            "left_eye_points": [],
            "right_eye_points": [],
            "left_eye_contour": [],
            "right_eye_contour": []
        }

        if not results.multi_face_landmarks:
            return detection_data

        detection_data["faces_detected"] = len(results.multi_face_landmarks)
        face_landmarks = results.multi_face_landmarks[0]

        try:
            # Get EAR calculation points
            left_eye_points = self.get_eye_landmarks(face_landmarks, image_rgb.shape, LEFT_EYE_INDICES)
            right_eye_points = self.get_eye_landmarks(face_landmarks, image_rgb.shape, RIGHT_EYE_INDICES)
            
            # Get complete contours for visualization
            left_eye_contour = self.get_eye_contour(face_landmarks, image_rgb.shape, LEFT_EYE_CONTOUR)
            right_eye_contour = self.get_eye_contour(face_landmarks, image_rgb.shape, RIGHT_EYE_CONTOUR)

            # Calculate EAR for both eyes
            ear_left = self.calculate_ear(left_eye_points)
            ear_right = self.calculate_ear(right_eye_points)
            ear_avg = (ear_left + ear_right) / 2.0

            # Determine eye state with smoothing
            current_state = "closed" if ear_avg < self.ear_threshold else "open"
            smoothed_state = self.smooth_eye_state(current_state)

            detection_data.update({
                "eyes_detected": True,
                "ear_left": round(ear_left, 3),
                "ear_right": round(ear_right, 3),
                "ear_avg": round(ear_avg, 3),
                "eye_state": smoothed_state,
                "is_drowsy": smoothed_state == "closed",
                "confidence": min(ear_avg * 3, 1.0),
                "left_eye_points": left_eye_points,
                "right_eye_points": right_eye_points,
                "left_eye_contour": left_eye_contour,
                "right_eye_contour": right_eye_contour
            })

        except Exception as e:
            logger.error(f"Eye detection error: {e}")

        return detection_data

# Global detector instance
detector = DrowsinessDetector()

def draw_detection_results(image, detection_data):
    """Draw eye landmarks and status on image"""
    # Draw eye contours
    for point in detection_data.get("left_eye_contour", []):
        cv2.circle(image, point, 1, (0, 255, 0), -1)
    for point in detection_data.get("right_eye_contour", []):
        cv2.circle(image, point, 1, (0, 255, 0), -1)
    
    # Draw EAR calculation points
    for point in detection_data.get("left_eye_points", []):
        cv2.circle(image, point, 3, (255, 0, 0), -1)
    for point in detection_data.get("right_eye_points", []):
        cv2.circle(image, point, 3, (255, 0, 0), -1)
    
    # Draw status text
    eye_state = detection_data.get("eye_state", "unknown")
    ear_avg = detection_data.get("ear_avg", 0)
    color = (0, 255, 0) if eye_state == "open" else (0, 0, 255)
    
    cv2.putText(image, f"State: {eye_state.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"EAR: {ear_avg:.3f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"Threshold: {detector.ear_threshold}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "drowsiness-detection",
        "timestamp": time.time(),
        "ear_threshold": detector.ear_threshold
    })

@app.route('/detect', methods=['POST'])
def detect_drowsiness():
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Update threshold if provided
        if 'ear_threshold' in data:
            detector.ear_threshold = float(data['ear_threshold'])
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Detect eyes
        detection_data = detector.detect_eyes(image_rgb)
        
        # Draw results on image
        annotated_image = draw_detection_results(image_np.copy(), detection_data)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        detection_data["annotated_image"] = base64.b64encode(buffer).decode('utf-8')
        detection_data["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Detection: {detection_data['eye_state']} (EAR: {detection_data['ear_avg']})")
        
        return jsonify(detection_data)
        
    except Exception as e:
        logger.error(f"Drowsiness detection error: {str(e)}")
        return jsonify({
            "error": f"Detection failed: {str(e)}",
            "eye_state": "unknown",
            "is_drowsy": False,
            "ear_avg": 0.0
        }), 500

@app.route('/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        if 'ear_threshold' in data:
            detector.ear_threshold = float(data['ear_threshold'])
            return jsonify({"message": f"EAR threshold updated to {detector.ear_threshold}"})
        return jsonify({"error": "No valid settings provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Accurate Drowsiness Detection Server...")
    logger.info(f"Initial EAR threshold: {detector.ear_threshold}")
    app.run(host='0.0.0.0', port=5001, debug=False)