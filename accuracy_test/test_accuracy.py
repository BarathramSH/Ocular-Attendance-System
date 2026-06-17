import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_model_accuracy():
    print("Loading AI Model...")
    
    # 1. Load your trained SVM model (Make sure the path matches where your .pkl is saved)
    # with open("models/svm_model.pkl", "rb") as file:
    #     model = pickle.load(file)
    
    # 2. Provide the "Ground Truth" (The actual names of the people in your test photos)
    # In a real scenario, you would load these from a testing folder.
    y_true = ["Barath", "Barath", "Stranger", "Barath", "Stranger"]
    
    # 3. Provide the "Predictions" (What your AI guessed when looking at those photos)
    # y_pred = model.predict(test_face_embeddings) 
    
    # --- FOR TESTING NOW: Let's use fake predictions so you can see how the math works ---
    y_pred = ["Barath", "Stranger", "Stranger", "Barath", "Barath"] 

    # 4. Generate the Report
    print("\n--- AI PERFORMANCE REPORT ---")
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_model_accuracy()