# app.py
import os
from flask import Flask, request, jsonify, render_template
from infer_utils import load_model, predict_topk

MODEL_PATH = os.environ.get("MODEL_PATH", "saved_model/classifier.keras")
model = load_model(MODEL_PATH)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file under form key 'file'"}), 400
    try:
        img_bytes = request.files["file"].read()
        topk = predict_topk(model, img_bytes, k=3)
        return jsonify({"predictions": topk})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
