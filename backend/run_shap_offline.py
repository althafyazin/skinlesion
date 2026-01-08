"""
Offline SHAP generation script.
Run this separately AFTER a user uploads an image via the UI.
"""

import os
import numpy as np
from PIL import Image

from model_loader import load_trained_model
from preprocess import preprocess_image
from explainability.shap_explainer import generate_shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGE_PATH = os.path.join(STATIC_DIR, "last_uploaded.jpg")

def main():
    if not os.path.exists(IMAGE_PATH):
        print("[ERROR] No uploaded image found.")
        print("Open the UI, upload an image, then run this script again.")
        return

    print("[INFO] Loading model...")
    model = load_trained_model()

    print("[INFO] Loading last uploaded image...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = preprocess_image(image)

    print("[INFO] Computing prediction index...")
    preds = model.predict([img_tensor, img_tensor])[0]
    pred_index = int(np.argmax(preds))

    print("[INFO] Running SHAP OFFLINE (this may take time)...")
    generate_shap(model, img_tensor, pred_index)

    print("[SUCCESS] SHAP image saved to backend/static/shap.png")

if __name__ == "__main__":
    main()
