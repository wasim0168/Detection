import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify
from collections import deque

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# smoothing window
prediction_buffer = deque(maxlen=10)

current_action = "Detecting..."

previous_center = None


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle



movement_history = deque(maxlen=15)

def detect_action(landmarks):

    global previous_center

    nose = [landmarks[0].x, landmarks[0].y]

    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]

    left_hip = [landmarks[23].x, landmarks[23].y]
    right_hip = [landmarks[24].x, landmarks[24].y]


    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )

    hip_center = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )


    # motion tracking
    movement = 0

    if previous_center is not None:

        movement = np.linalg.norm(
            np.array(hip_center) -
            np.array(previous_center)
        )

    previous_center = hip_center


    movement_history.append(movement)

    avg_movement = sum(movement_history) / len(movement_history)


    # torso angle detection
    spine_angle = calculate_angle(
        shoulder_center,
        hip_center,
        (hip_center[0], hip_center[1] + 0.2)
    )


    head_drop = nose[1] - shoulder_center[1]


    # FINAL CLASSIFICATION LOGIC


    if avg_movement > 0.035:
        return "Walking"


    if abs(shoulder_center[1] - hip_center[1]) > 0.18:
       return "Standing"


    if head_drop > 0.15 and spine_angle > 40:
      return "Sleeping"


    return "Sitting"


def smooth_prediction(prediction):

    prediction_buffer.append(prediction)

    return max(
        set(prediction_buffer),
        key=prediction_buffer.count
    )



def generate_frames():

    global current_action

    while True:

        success, frame = cap.read()

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose.process(rgb)

        if result.pose_landmarks:

            prediction = detect_action(
                result.pose_landmarks.landmark
            )

            current_action = smooth_prediction(prediction)

            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"Action: {current_action}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        _, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/action')
def action():

    return jsonify({
        "action": current_action
    })


@app.route('/')
def home():

    return "Advanced Action Detection Running"


if __name__ == '__main__':

    app.run(debug=True, port=5002)
