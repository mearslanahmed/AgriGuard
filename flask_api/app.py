import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

print("Loading disease model...")
MODEL_PATH = os.path.join("model", "best_model.keras")
CLASS_INDICES_PATH = os.path.join("model", "class_indices.json")

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

print(f"Disease model loaded. {len(class_indices)} classes ready.")

print("Loading plant gate model (EfficientNetB0)...")
gate_model = EfficientNetB0(weights="imagenet")
print("Plant gate ready.")

# ImageNet class names that are plant/leaf related
# These are the WordNet IDs for plant-related ImageNet classes
PLANT_KEYWORDS = [
    "leaf", "plant", "flower", "tree", "herb", "vegetable",
    "fungus", "moss", "fern", "shrub", "bud", "petal",
    "daisy", "mushroom", "cabbage", "broccoli", "cauliflower",
    "corn", "ear", "spike", "stalk", "vine", "fruit",
    "tomato", "pepper", "potato", "cucumber", "zucchini",
    "lettuce", "spinach", "artichoke", "lemon", "orange",
    "banana", "strawberry", "fig", "pineapple", "pomegranate",
]

def is_plant_image(img_pil):
    """
    Uses EfficientNetB0 pretrained on ImageNet to check
    if the image contains a plant or leaf.
    Returns (is_plant: bool, reason: str)
    """
    # EfficientNetB0 expects 224x224
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet specific preprocessing

    preds = gate_model.predict(img_array, verbose=0)
    # decode_predictions returns top-5 [(class_id, class_name, confidence), ...]
    top_preds = decode_predictions(preds, top=5)[0]

    for (_, class_name, confidence) in top_preds:
        class_name_lower = class_name.lower()
        for keyword in PLANT_KEYWORDS:
            if keyword in class_name_lower:
                return True, f"Detected: {class_name} ({confidence*100:.1f}%)"

    # Log what it actually saw for debugging
    top_labels = [f"{name}({conf*100:.1f}%)" for (_, name, conf) in top_preds]
    return False, f"Not a plant. Top predictions: {', '.join(top_labels)}"


def parse_class_name(class_name):
    parts = class_name.replace("__", "_").split("_")
    crop = parts[0]
    if crop == "Pepper":
        crop = "Pepper Bell"
    disease_parts = parts[1:] if crop != "Pepper Bell" else parts[2:]
    disease = " ".join(disease_parts).replace("  ", " ").strip()
    if disease.lower() == "healthy":
        disease = "Healthy"
    return crop, disease


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Gate check — reject non-plant images before disease model
        is_plant, reason = is_plant_image(img)
        if not is_plant:
            return jsonify({
                "success": False,
                "error": "The image does not appear to contain a crop leaf. Please upload a clear photo of a plant leaf.",
                "debug": reason
            }), 422

        # Passed gate — run disease model
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        class_name = class_indices[str(predicted_index)]
        crop, disease = parse_class_name(class_name)

        return jsonify({
            "success": True,
            "crop": crop,
            "disease": disease,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
            "is_healthy": disease.lower() == "healthy"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)