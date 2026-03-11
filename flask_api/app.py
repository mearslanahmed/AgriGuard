import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load model and class indices once at startup
# I will load these once so every request doesn't reload them
# Loading a model takes ~2s, I don't want that per request

print("Loading model...")
MODEL_PATH = os.path.join("model", "best_model.keras")
CLASS_INDICES_PATH = os.path.join("model", "class_indices.json")

model = load_model(MODEL_PATH)

with open (CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)    # {0: 'Tomato_healthy', 1: '...', ...}

print(f"Model loaded. {len(class_indices)} classes ready.")

# --- Helper: parse class name into crop and disease ---
def parse_class_name(class_name):
    """
    Converts raw class name into readable crop and disease.
    Example: 'Tomato_Early_blight' -> crop='Tomato',disease='Early Blight'
    """

    parts = class_name.replace("__", "_").split("_")  # Handle double underscores if any

    # First part is always the crop
    crop = parts[0]
    if crop == "Pepper":
        crop = "Pepper Bell"

    # Rest is the disease
    disease_parts = parts[1:] if crop != "Pepper Bell" else parts[2:]
    disease = " ".join(disease_parts).replace("  ", " ").strip()

    # Clean up healthy case
    if disease.lower() == "healthy":
        disease = "Healthy"

    return crop, disease


# --- Health check endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if an image was sent
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Please upload an image as form-data with key 'image'"}), 400
    
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename provided."}), 400
    
    try:
        # 2. Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # ensure 3 channels
        img = img.resize((224, 224))  # Resize to model's expected input - MobileNetV2 expects 224x224

        # Convert to numpy array and normalize (same as training)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)


        # 3. Run Prediction
        predictions = model.predict(img_array, verbose=0)   # shape: (1,15)
        predicted_index = int(np.argmax(predictions[0]))    # index of highest probability
        confidence = float(np.max(predictions[0]))          # highest probability value


        # 4. Map index to class name
        class_name = class_indices[str(predicted_index)]
        crop, disease = parse_class_name(class_name)


        # 5. Build response
        return jsonify({
            "success": True,
            "crop": crop,
            "disease": disease,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),  # as percentage
            "is_healthy": disease.lower() == "healthy"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)