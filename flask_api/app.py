import os
import json
import numpy as np
import io
from flask import Flask, request, jsonify
from PIL import Image

# Import core Keras engine layers
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Enforce a strict file-upload limit (5MB max) to protect system memory
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'model')


# LOAD CORE APPLICATION ARTIFACTS WITH VERIFIED FILENAMES

print("Loading crop classifier (Model 1)...")
model1 = load_model(os.path.join(MODEL_DIR, 'AgriGuard_Model1_Final.keras'))

print("Loading disease classifier (Model 2)...")
model2 = load_model(os.path.join(MODEL_DIR, 'AgriGuard_Model2_Final.keras'))

with open(os.path.join(MODEL_DIR, 'model1_crop_mapping.json')) as f:
    idx_to_crop = json.load(f)

with open(os.path.join(MODEL_DIR, 'model2_disease_mapping.json')) as f:
    idx_to_disease = json.load(f)

# Pre-build our global cross-crop disease evaluation indexes
crop_disease_map = {}
for idx_str, label in idx_to_disease.items():
    if '(' in label and label.endswith(')'):
        crop = label[label.rfind('(') + 1:-1]
        crop_disease_map.setdefault(crop, []).append((int(idx_str), label))


def preprocess(image_bytes):
    """
    Standardizes incoming byte arrays for EfficientNetB3 inference.
    Aggressive chroma filtering removed to protect chlorotic/dried crop signatures.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_resized = img.resize((300, 300))
    arr = np.array(img_resized) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        image_bytes = file.read()
        tensor = preprocess(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    try:
        
        # STAGE 1: CROP SELECTION
        
        crop_probs = model1.predict(tensor, verbose=0)[0]
        crop_idx = int(np.argmax(crop_probs))
        crop_conf = float(crop_probs[crop_idx])
        crop_name = idx_to_crop[str(crop_idx)]

        # Get the runner-up crop guess in case of close ambiguity (e.g., Mango vs Tomato)
        sorted_crop_indices = np.argsort(crop_probs)[::-1]
        runner_up_idx = int(sorted_crop_indices[1])
        runner_up_name = idx_to_crop[str(runner_up_idx)]

        # Optimized Threshold: Adjusted to 0.45 to support field conditions.
        # Native 'Unknown' class handles non-plant noise filters.
        if crop_conf < 0.45 or crop_name == 'Unknown':
            return jsonify({
                'success': False,
                'error': 'The plant leaf is not supported by AgriGuard, or the image is too blurry. Please try again.',
                'crop': 'Unknown',
                'crop_confidence': round(crop_conf, 4)
            }), 422

        
        # STAGE 2: DISEASE EVALUATION
        
        disease_probs = model2.predict(tensor, verbose=0)[0]

        # Pull candidate diseases for the primary crop guess
        candidates = crop_disease_map.get(crop_name, [])

        if candidates:
            best_primary_idx, best_primary_label = max(candidates, key=lambda x: disease_probs[x[0]])
            primary_disease_conf = float(disease_probs[best_primary_idx])
        else:
            best_primary_idx = int(np.argmax(disease_probs))
            best_primary_label = idx_to_disease[str(best_primary_idx)]
            primary_disease_conf = float(disease_probs[best_primary_idx])

        # Dynamic Fallback: If primary crop yields a poor disease match (< 30%),
        # but the runner-up crop guess has candidate diseases, check the runner-up instead.
        if primary_disease_conf < 0.30 and crop_probs[runner_up_idx] > 0.20:
            runner_candidates = crop_disease_map.get(runner_up_name, [])
            if runner_candidates:
                best_runner_idx, best_runner_label = max(runner_candidates, key=lambda x: disease_probs[x[0]])
                runner_disease_conf = float(disease_probs[best_runner_idx])

                # If the runner-up crop gives a much better disease diagnosis, swap to it!
                if runner_disease_conf > primary_disease_conf:
                    crop_name = runner_up_name
                    crop_conf = float(crop_probs[runner_up_idx])
                    best_primary_idx = best_runner_idx
                    best_primary_label = best_runner_label
                    primary_disease_conf = runner_disease_conf

        # Optimized Disease Gate: Lowered to 0.35 to catch early/subtle infections
        if primary_disease_conf < 0.35:
            return jsonify({
                'success': False,
                'error': 'The model recognizes the crop structure but cannot reliably diagnose any disease anomalies. Ensure proper lighting.',
                'crop': crop_name,
                'crop_confidence': round(crop_conf, 4)
            }), 422

        # Formatting strings for mobile UI view
        disease_name = best_primary_label[:best_primary_label.rfind('(')].strip() if '(' in best_primary_label else best_primary_label
        is_healthy = 'healthy' in best_primary_label.lower()

        return jsonify({
            'success': True,
            'crop': crop_name.strip(),
            'crop_confidence': round(crop_conf, 4),
            'disease': disease_name.strip(),
            'disease_label': best_primary_label.strip(), # Matches MongoDB values exactly
            'disease_confidence': round(primary_disease_conf, 4),
            'confidence': round(primary_disease_conf * 100, 2),
            'is_healthy': is_healthy
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)