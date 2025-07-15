from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import onnxruntime as rt
import os
import io
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) #Edit this http://localhost:5173 to the site on which your project is hosted

# Constants
IMG_SIZE = (224, 224)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Class names
class_names = {
    0: "Blight",
    1: "Common_Rust",
    2: "Gray_Leaf_Spot",
    3: "Healthy",
    4: "MLN",
    5: "FAW",
    6: "Maize_Streak"
}

# Load ONNX model
model_path = "Model/model_0_.onnx"
sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def save_uploaded_image(image_bytes, filename):
    try:
        ext = filename.split('.')[-1] if '.' in filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return file_path if os.path.exists(file_path) else None
    except Exception as e:
        print("Error saving image:", e)
        return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        image_bytes = file.read()
        image_path = save_uploaded_image(image_bytes, file.filename)
        if not image_path:
            return jsonify({"error": "Failed to save uploaded image"}), 500

        filename_only = os.path.basename(image_path)
        input_image = preprocess_image(image_bytes)
        predictions = sess.run(None, {input_name: input_image})
        
        predicted_index = int(np.argmax(predictions[0], axis=1)[0])
        confidence = float(np.max(predictions[0]))

        raw_class = class_names.get(predicted_index, "Unknown")
        predicted_class = raw_class.replace("_", " ").title()

        # Example hardcoded response (remove if dynamic data added)
        dummy_info = {
            "category": "Fungal",
            "part_scanned": "Leaf",
            "symptoms": "Visible spots",
            "treatment": "Use fungicide",
            "prevention": "Crop rotation"
        }

        response = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "category": dummy_info["category"],
            "part_scanned": dummy_info["part_scanned"],
            "symptoms": dummy_info["symptoms"],
            "treatment": dummy_info["treatment"],
            "prevention": dummy_info["prevention"],
            "image_filename": filename_only,
            "original_filename": file.filename,
            "timestamp": datetime.utcnow().isoformat()
        }

        return jsonify(response), 200

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
