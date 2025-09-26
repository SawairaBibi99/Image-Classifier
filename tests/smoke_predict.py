# tests/smoke_predict.py
import requests, sys

img_path = sys.argv[1]
with open(img_path, "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
print(r.status_code, r.json())
