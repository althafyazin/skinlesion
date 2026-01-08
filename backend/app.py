import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np

# Optional: reduce TF log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from model_loader import load_trained_model
from preprocess import preprocess_image
from predict import predict_image
from explainability.gradcam import make_gradcam_heatmap, save_gradcam
from explainability.lime_explainer import generate_lime

app = Flask(__name__)

# Load model once
model = load_trained_model()
LAST_CONV_LAYER = "block14_sepconv2_act"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LAST_IMAGE_PATH = os.path.join(STATIC_DIR, "last_uploaded.jpg")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # -------- Read uploaded image --------
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        # -------- Save image for OFFLINE SHAP --------
        os.makedirs(STATIC_DIR, exist_ok=True)
        image.save(LAST_IMAGE_PATH)

        # -------- Preprocess & Predict --------
        img_tensor = preprocess_image(image)
        predicted_class, confidence, probabilities, pred_index = predict_image(
            model, img_tensor
        )

        # -------- Grad-CAM --------
        heatmap = make_gradcam_heatmap(
            img_tensor, model, LAST_CONV_LAYER, pred_index
        )
        resized = np.array(image.resize((299, 299)))
        save_gradcam(resized, heatmap)

        # -------- LIME (near real-time) --------
        generate_lime(resized, model, pred_index)

        return render_template(
            "index.html",
            show_result=True,
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            probabilities=probabilities
        )

    return render_template("index.html", show_result=False)

if __name__ == "__main__":
    app.run(debug=True)
