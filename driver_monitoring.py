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
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.1)
    else:
        import os
        os.system('play -nq -t alsa synth 0.3 sine 1000')

def trigger_alarm():
    if not alarm_active[0]:
        alarm_active[0] = True
        threading.Thread(target=lambda: (play_alarm(), alarm_active._setitem_(0, False))).start()

alarm_active = [False]

# ==== Mediapipe setup ====
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

STEERING_WHEEL_Y_MIN = 0.35
STEERING_WHEEL_Y_MAX = 0.75
STEERING_WHEEL_X_MIN = 0.2
STEERING_WHEEL_X_MAX = 0.8

EYE_CLOSED_FRAMES = 0
EYE_CLOSED_THRESHOLD = 15

# ==== HELPER FUNCTIONS ====
def detect_posture(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    dx = nose.x - shoulder_center_x
    dy = nose.y - shoulder_center_y

    if dy < -0.05:
        return "Leaning Forward"
    elif dy > 0.1:
        return "Leaning Backward"
    elif dx > 0.1:
        return "Leaning Right"
    elif dx < -0.1:
        return "Leaning Left"
    else:
        return "Upright"

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

def eye_aspect_ratio(face_landmarks, eye_indices, img_w, img_h):
    p2 = np.array([face_landmarks.landmark[eye_indices[1]].x * img_w, face_landmarks.landmark[eye_indices[1]].y * img_h])
    p6 = np.array([face_landmarks.landmark[eye_indices[5]].x * img_w, face_landmarks.landmark[eye_indices[5]].y * img_h])
    p3 = np.array([face_landmarks.landmark[eye_indices[2]].x * img_w, face_landmarks.landmark[eye_indices[2]].y * img_h])
    p5 = np.array([face_landmarks.landmark[eye_indices[4]].x * img_w, face_landmarks.landmark[eye_indices[4]].y * img_h])
    p1 = np.array([face_landmarks.landmark[eye_indices[0]].x * img_w, face_landmarks.landmark[eye_indices[0]].y * img_h])
    p4 = np.array([face_landmarks.landmark[eye_indices[3]].x * img_w, face_landmarks.landmark[eye_indices[3]].y * img_h])
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal)

def detect_seatbelt(frame, face_box):
    x1, y1, x2, y2 = face_box
    roi_y1 = y2 + 10
    roi_y2 = min(frame.shape[0], roi_y1 + 150)
    roi_x1 = max(0, x1 - 50)
    roi_x2 = min(frame.shape[1], x2 + 50)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return False, roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=15)
    seatbelt_detected = False
    if lines is not None:
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l))
            if 30 < abs(angle) < 70:
                seatbelt_detected = True
                cv2.line(roi, (x1_l, y1_l), (x2_l, y2_l), (255, 0, 0), 2)
    return seatbelt_detected, roi

def detect_emotion(face_landmarks, img_w, img_h):
    left_mouth = np.array([face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h])
    right_mouth = np.array([face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h])
    top_mouth = np.array([face_landmarks.landmark[13].x * img_w, face_landmarks.landmark[13].y * img_h])
    bottom_mouth = np.array([face_landmarks.landmark[14].x * img_w, face_landmarks.landmark[14].y * img_h])

    mouth_open = np.linalg.norm(top_mouth - bottom_mouth)
    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    mouth_ratio = mouth_open / (mouth_width + 1e-6)

    smile_curve = ((left_mouth[1] + right_mouth[1]) / 2) - top_mouth[1]

    if smile_curve > 5 and mouth_ratio > 0.25:
        return "Happy"
    elif smile_curve < -3 and mouth_ratio < 0.3:
        return "Sad"
    else:
        return "Neutral"

# ==== MAIN FUNCTION ====
def driver_monitoring():
    global EYE_CLOSED_FRAMES
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam!")
        return

    print("\n🚗 Starting Driver Monitoring...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)
        results_face = face_mesh.process(rgb_frame)
        results_hands = hands.process(rgb_frame)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        driver_status = []
        face_box = None

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            posture = detect_posture(results_pose.pose_landmarks.landmark)
            hands_status = detect_hands_on_wheel(results_pose.pose_landmarks.landmark)
            driver_status.append(f"Posture: {posture}")
            driver_status.append(f"Hands: {hands_status}")

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

                x_coords = [int(lm.x * img_w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * img_h) for lm in face_landmarks.landmark]
                face_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)

                left_eye, right_eye = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
                left_EAR = eye_aspect_ratio(face_landmarks, left_eye, img_w, img_h)
                right_EAR = eye_aspect_ratio(face_landmarks, right_eye, img_w, img_h)
                avg_EAR = (left_EAR + right_EAR) / 2.0
                if avg_EAR < 0.21:
                    EYE_CLOSED_FRAMES += 1
                else:
                    EYE_CLOSED_FRAMES = 0
                if EYE_CLOSED_FRAMES >= EYE_CLOSED_THRESHOLD:
                    driver_status.append("Drowsy / Eyes closed")

                mouth_open = np.linalg.norm(np.array([face_landmarks.landmark[13].x * img_w, face_landmarks.landmark[13].y * img_h]) -
                                            np.array([face_landmarks.landmark[14].x * img_w, face_landmarks.landmark[14].y * img_h]))
                mouth_width = np.linalg.norm(np.array([face_landmarks.landmark[78].x * img_w, face_landmarks.landmark[78].y * img_h]) -
                                             np.array([face_landmarks.landmark[308].x * img_w, face_landmarks.landmark[308].y * img_h]))
                if (mouth_open / mouth_width) > 0.6:
                    driver_status.append("Yawning")

                emotion = detect_emotion(face_landmarks, img_w, img_h)
                driver_status.append(f"Emotion: {emotion}")

        if results_hands.multi_hand_landmarks and face_box:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    if (face_box[0] <= x <= face_box[2]) and (face_box[1] <= y <= face_box[3]):
                        driver_status.append("Possible phone use")
                        break

        if face_box:
            seatbelt_detected, roi = detect_seatbelt(frame, face_box)
            if seatbelt_detected:
                driver_status.append("✅ Seatbelt detected")
            else:
                driver_status.append("⚠ Seatbelt not fastened")
            cv2.imshow("Seatbelt ROI", roi)

        for i, status in enumerate(driver_status):
            cv2.putText(frame, status, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if any(alert in " ".join(driver_status) for alert in ["Drowsy", "Yawning", "Possible phone use",
                                                              "Seatbelt not fastened", "No hands on wheel"]):
            trigger_alarm()

        cv2.imshow('Driver Monitoring', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            print("Exiting Monitoring...\n")
            break

    cap.release()
    cv2.destroyAllWindows()