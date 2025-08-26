import cv2
from deepface import DeepFace

# Path للصورة المرجعية
reference_img = "C:\\Users\\User\\Desktop\\BS\\project3\\WhatsApp Image 2024-07-10 at 20.07.17_20b3910c.jpg"

def face_verification():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    verified = False
    result_text = "Analyzing..."
    color = (255, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 10 == 0:
            try:
                result = DeepFace.verify(
                    img1_path=reference_img,
                    img2_path=frame,
                    enforce_detection=False
                )

                if result["verified"]:
                    result_text = "Welcome Rawan"
                    color = (0, 255, 0)
                    verified = True
                    break
                else:
                    result_text = "Unknown"
                    color = (0, 0, 255)

            except Exception as e:
                print("Error:", e)
                result_text = "Error"
                color = (0, 0, 255)

        cv2.putText(frame, result_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Face Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return verified