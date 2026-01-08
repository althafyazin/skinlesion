import os
from explainability.shap_explainer import generate_shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATUS_FILE = os.path.join(STATIC_DIR, "shap_status.txt")

def run_shap_in_background(model, img_tensor, pred_index):
    """
    Runs SHAP in background thread
    """
    os.makedirs(STATIC_DIR, exist_ok=True)

    # mark SHAP as running
    with open(STATUS_FILE, "w") as f:
        f.write("running")

    try:
        generate_shap(model, img_tensor, pred_index)

        # mark SHAP as done
        with open(STATUS_FILE, "w") as f:
            f.write("done")

    except Exception as e:
        with open(STATUS_FILE, "w") as f:
            f.write(f"error: {str(e)}")
