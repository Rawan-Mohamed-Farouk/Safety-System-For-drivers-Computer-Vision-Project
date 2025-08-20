import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import platform

# ==== SOUND SETUP ====
def play_alarm():
    if platform.system() == "Windows":
        import winsound
        for _ in range(3):  # 3 beeps
            winsound.Beep(1000, 300)  # freq=1000Hz, duration=300ms
            time.sleep(0.1)
    else:
        # cross-platform fallback
        import os
        os.system('play -nq -t alsa synth 0.3 sine 1000')

def trigger_alarm():
    if not alarm_active[0]:  # avoid overlapping alarms
        alarm_active[0] = True
        threading.Thread(target=lambda: (play_alarm(), alarm_active.__setitem__(0, False))).start()

alarm_active = [False]

# ==== Mediapipe setup ====
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize models
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Steering wheel region (approximate) - normalized coords (update based on camera position)
STEERING_WHEEL_Y_MIN = 0.35
STEERING_WHEEL_Y_MAX = 0.75
STEERING_WHEEL_X_MIN = 0.2
STEERING_WHEEL_X_MAX = 0.8

# Eye closure tracking
EYE_CLOSED_FRAMES = 0
EYE_CLOSED_THRESHOLD = 15  # frames (~0.5s at 30 FPS)

# ==== HELPER FUNCTIONS ====
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

# Eye aspect ratio function
def eye_aspect_ratio(face_landmarks, eye_indices, img_w, img_h):
    p2 = np.array([face_landmarks.landmark[eye_indices[1]].x * img_w,
                   face_landmarks.landmark[eye_indices[1]].y * img_h])
    p6 = np.array([face_landmarks.landmark[eye_indices[5]].x * img_w,
                   face_landmarks.landmark[eye_indices[5]].y * img_h])
    p3 = np.array([face_landmarks.landmark[eye_indices[2]].x * img_w,
                   face_landmarks.landmark[eye_indices[2]].y * img_h])
    p5 = np.array([face_landmarks.landmark[eye_indices[4]].x * img_w,
                   face_landmarks.landmark[eye_indices[4]].y * img_h])
    p1 = np.array([face_landmarks.landmark[eye_indices[0]].x * img_w,
                   face_landmarks.landmark[eye_indices[0]].y * img_h])
    p4 = np.array([face_landmarks.landmark[eye_indices[3]].x * img_w,
                   face_landmarks.landmark[eye_indices[3]].y * img_h])

    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal)

# Seatbelt Detection using HoughLines
def detect_seatbelt(frame, face_box):
    x1, y1, x2, y2 = face_box
    roi_y1 = y2 + 10  # just below face
    roi_y2 = min(frame.shape[0], roi_y1 + 150)  # chest region
    roi_x1 = max(0, x1 - 50)
    roi_x2 = min(frame.shape[1], x2 + 50)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return False, roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=40, maxLineGap=15)

    seatbelt_detected = False
    if lines is not None:
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l))
            if 30 < abs(angle) < 70:  # typical seatbelt diagonal angle
                seatbelt_detected = True
                cv2.line(roi, (x1_l, y1_l), (x2_l, y2_l), (255, 0, 0), 2)

    return seatbelt_detected, roi

# ==== MAIN PROGRAM ====
# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam!")
    exit()
print('\nCapturing...\n')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip horizontally so left/right are correct
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    
    # Convert BGR to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process all models
    results_pose = pose.process(rgb_frame)
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    
    # Convert back to BGR for display
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    driver_status = []
    face_box = None
    
    # Process pose results
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        posture = detect_posture(results_pose.pose_landmarks.landmark)
        hands_status = detect_hands_on_wheel(results_pose.pose_landmarks.landmark)
        
        driver_status.append(f"Posture: {posture}")
        driver_status.append(f"Hands: {hands_status}")
        
        # Draw steering wheel region for debugging
        x1, y1 = int(STEERING_WHEEL_X_MIN * img_w), int(STEERING_WHEEL_Y_MIN * img_h)
        x2, y2 = int(STEERING_WHEEL_X_MAX * img_w), int(STEERING_WHEEL_Y_MAX * img_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Process face results
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
            
            # Head pose estimation
            face_2d = []
            face_3d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]]
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                face_3d, face_2d, camera_matrix, dist_coeffs
            )
            rmat = cv2.Rodrigues(rotation_vector)[0]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360  # Pitch
            y_angle = angles[1] * 360  # Yaw

            if abs(y_angle) > 15:
                driver_status.append("Head turned away")
            elif x_angle < -7 or x_angle > 20:
                driver_status.append("Head tilted")
            else:
                driver_status.append("Head straight")

            # Face bounding box
            x_coords = [int(lm.x * img_w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * img_h) for lm in face_landmarks.landmark]
            face_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            cv2.rectangle(frame, (face_box[0], face_box[1]),
                          (face_box[2], face_box[3]), (0, 255, 0), 2)

            # Drowsiness detection
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            left_EAR = eye_aspect_ratio(face_landmarks, left_eye, img_w, img_h)
            right_EAR = eye_aspect_ratio(face_landmarks, right_eye, img_w, img_h)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            if avg_EAR < 0.21:
                EYE_CLOSED_FRAMES += 1
            else:
                EYE_CLOSED_FRAMES = 0

            if EYE_CLOSED_FRAMES >= EYE_CLOSED_THRESHOLD:
                driver_status.append("Drowsy / Eyes closed")

            # Yawning detection
            mouth_open = np.linalg.norm(
                np.array([face_landmarks.landmark[13].x * img_w, face_landmarks.landmark[13].y * img_h]) -
                np.array([face_landmarks.landmark[14].x * img_w, face_landmarks.landmark[14].y * img_h])
            )
            mouth_width = np.linalg.norm(
                np.array([face_landmarks.landmark[78].x * img_w, face_landmarks.landmark[78].y * img_h]) -
                np.array([face_landmarks.landmark[308].x * img_w, face_landmarks.landmark[308].y * img_h])
            )
            if (mouth_open / mouth_width) > 0.6:
                driver_status.append("Yawning")

    # Phone use detection
    if results_hands.multi_hand_landmarks and face_box:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if (face_box[0] <= x <= face_box[2]) and (face_box[1] <= y <= face_box[3]):
                    driver_status.append("Possible phone use")
                    break

    # Seatbelt detection
    if face_box:
        seatbelt_detected, roi = detect_seatbelt(frame, face_box)
        if seatbelt_detected:
            driver_status.append("✅ Seatbelt detected")
        else:
            driver_status.append("⚠ Seatbelt not fastened")
        cv2.imshow("Seatbelt ROI", roi)

    # Display all status information
    for i, status in enumerate(driver_status):
        cv2.putText(frame, status, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Trigger alarm if any critical warning
    if any(alert in " ".join(driver_status) for alert in ["Drowsy", "Yawning", "Head turned away",
                                                          "Head tilted", "Possible phone use",
                                                          "Seatbelt not fastened", "No hands on wheel"]):
        trigger_alarm()

    # Show video
    cv2.imshow('Driver Monitoring', frame)

    # Exit on ESC
    if cv2.waitKey(5) & 0xFF == 27:
        print('Exiting!\n')
        break

cap.release()
cv2.destroyAllWindows()