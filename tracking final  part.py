import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Steering wheel region (approximate) - normalized coords (update based on camera position)
STEERING_WHEEL_Y_MIN = 0.35
STEERING_WHEEL_Y_MAX = 0.75
STEERING_WHEEL_X_MIN = 0.2
STEERING_WHEEL_X_MAX = 0.8

def detect_posture(landmarks):
    # Extract important landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Compute shoulder center
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

    # Posture detection based on relative nose position
    dx = nose.x - shoulder_center_x
    dy = nose.y - shoulder_center_y

    if dy < -0.05:
        posture = "Leaning Forward"
    elif dy > 0.1:
        posture = "Leaning Backward"
    elif dx > 0.1:
        posture = "Leaning Right"
    elif dx < -0.1:
        posture = "Leaning Left"
    else:
        posture = "Upright"
    
    return posture

def detect_hands_on_wheel(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    def in_wheel_region(wrist):
        return (STEERING_WHEEL_X_MIN < wrist.x < STEERING_WHEEL_X_MAX and
                STEERING_WHEEL_Y_MIN < wrist.y < STEERING_WHEEL_Y_MAX)

    left_on_wheel = in_wheel_region(left_wrist)
    right_on_wheel = in_wheel_region(right_wrist)

    if left_on_wheel and right_on_wheel:
        return "Both hands on wheel"
    elif left_on_wheel or right_on_wheel:
        return "One hand on wheel"
    else:
        return "No hands on wheel"

# Start video capture (opens webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip horizontally so left/right are correct
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        posture = detect_posture(results.pose_landmarks.landmark)
        hands_status = detect_hands_on_wheel(results.pose_landmarks.landmark)

        cv2.putText(frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Hands: {hands_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Draw steering wheel region for debugging
        h, w, _ = frame.shape
        x1, y1 = int(STEERING_WHEEL_X_MIN * w), int(STEERING_WHEEL_Y_MIN * h)
        x2, y2 = int(STEERING_WHEEL_X_MAX * w), int(STEERING_WHEEL_Y_MAX * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Driver Monitoring', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
