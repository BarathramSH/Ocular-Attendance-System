import os

import cv2
from retinaface import RetinaFace

# ================= CONFIGURATION =================
STUDENT_NAME = input("Enter student's name: ").strip().lower()
MAX_IMAGES = 40
IMG_SIZE = 160
PADDING = 10
# =================================================

# --------- FIXED PROJECT PATH ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", STUDENT_NAME)
os.makedirs(DATASET_PATH, exist_ok=True)

print(f"\n📁 Saving images to: {DATASET_PATH}")

# --------- Initialize Webcam ---------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    exit()

count = len(os.listdir(DATASET_PATH))

print("\n--- Manual Face Capture ---")
print("Press 's' → Save image")
print("Press 'q' → Quit")
print(f"Target: {MAX_IMAGES} images\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # ================= THE HARDWARE FIX =================
    # This physically rotates the upside-down camera feed 180 degrees
    frame = cv2.flip(frame, 1)
    # ====================================================

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved: {count}/{MAX_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Face Capture - Press 's' to Save", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # ================= SAVE LOGIC =================
    if key == ord('s'):
        print("🔍 Detecting face...")
        faces = RetinaFace.detect_faces(frame)

        if isinstance(faces, dict):
            # --------- Choose Largest Face ---------
            largest_face = None
            max_area = 0

            for face in faces.values():
                x1, y1, x2, y2 = face["facial_area"]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_face = face

            # --------- DIRECT CROP (No complex math rotation!) ---------
            x1, y1, x2, y2 = largest_face["facial_area"]
            h, w, _ = frame.shape
            
            # Add padding safely
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w, x2 + PADDING)
            y2 = min(h, y2 + PADDING)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                file_path = os.path.join(DATASET_PATH, f"{STUDENT_NAME}_{count}.jpg")
                cv2.imwrite(file_path, face_crop)
                count += 1
                print(f"✅ Saved clean image {count}")
            else:
                print("⚠ Face crop failed. Try again.")
        else:
            print("⚠ No face detected. Adjust position.")

    # Quit
    if key == ord('q') or count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Face collection completed!")