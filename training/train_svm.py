import pickle

import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import os

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "svm", "svm_face_model.pkl")
LABEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "svm", "label_encoder.pkl")
# =================================================

print("🚀 Initializing FaceNet Extractor...")
embedder = FaceNet()

X = []  # embeddings
y = []  # labels (student names)

print(f"📁 Scanning dataset folder: {DATASET_PATH}")

# ================= DATASET LOOP =================
for student_name in os.listdir(DATASET_PATH):

    student_dir = os.path.join(DATASET_PATH, student_name)

    if not os.path.isdir(student_dir):
        continue

    print(f"🔄 Processing images for: {student_name}")

    for image_name in os.listdir(student_dir):

        image_path = os.path.join(student_dir, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        # Convert BGR → RGB (required for FaceNet)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Expand dimensions
        img_rgb = np.expand_dims(img_rgb, axis=0)

        # Extract embedding
        embedding = embedder.embeddings(img_rgb)[0]

        X.append(embedding)
        y.append(student_name)

# ================= DATA CHECK =================

# Convert to numpy array
X = np.array(X)

print(f"\n📊 Total images processed: {len(X)}")
print(f"👥 Total students: {len(set(y))}")

if len(X) == 0:
    print("❌ No embeddings extracted. Check dataset.")
    exit()

# ================= LABEL ENCODING =================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ================= TRAIN SVM =================

print("\n🧠 Training the SVM Classifier...")

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X, y_encoded)

# ================= SAVE MODELS =================

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(svm_model, f)

with open(LABEL_SAVE_PATH, 'wb') as f:
    pickle.dump(encoder, f)

print("\n✅ Training Complete!")
print(f"📦 Models saved to: {os.path.dirname(MODEL_SAVE_PATH)}")