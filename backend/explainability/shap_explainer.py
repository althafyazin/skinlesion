import os
import shap
import numpy as np
import matplotlib.pyplot as plt

# absolute path to backend/static
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

def generate_shap(model, img_tensor, pred_index):
    """
    SHAP explanation for a TWO-INPUT Keras ensemble model
    """

    # small background set (must match input shape)
    background = np.random.random((10, 299, 299, 3))

    # IMPORTANT: pass the MODEL, not a function
    explainer = shap.GradientExplainer(
        model,
        [background, background]   # two inputs
    )

    # IMPORTANT: pass inputs as a list
    shap_values = explainer.shap_values(
        [img_tensor, img_tensor]
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    path = os.path.join(STATIC_DIR, "shap.png")

    # shap_values is a list per class → use predicted class
    shap.image_plot(
        shap_values[pred_index],
        img_tensor,
        show=False
    )

    plt.savefig(path, bbox_inches="tight")
    plt.close()
