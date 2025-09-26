# infer_utils.py
import io
import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

IMG_SIZE = 96  # must match train.py

def load_model(model_path="saved_model/classifier.keras"):
    return keras.models.load_model(model_path)

def _preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    # same preprocessing as training (MobileNetV2)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)  # batch dim
    return x

def predict_topk(model, image_bytes: bytes, k: int = 3):
    x = _preprocess_image_bytes(image_bytes)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(-probs)[:k]
    return [{"class": CLASS_NAMES[i], "prob": float(probs[i])} for i in idx]
