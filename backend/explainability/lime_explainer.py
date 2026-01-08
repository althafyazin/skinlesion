import os
from lime import lime_image
import matplotlib.pyplot as plt

# absolute path to backend/static
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

def generate_lime(image, model, pred_index):
    def predict_wrapper(images):
        return model.predict([images, images])

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image,
        predict_wrapper,
        top_labels=5,      # IMPORTANT: avoid label mismatch
        hide_color=0,
        num_samples=200    # reduced for sanity
    )

    # always safe label
    label = explanation.top_labels[0]

    lime_img, _ = explanation.get_image_and_mask(
        label,
        positive_only=True,
        hide_rest=False
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    path = os.path.join(STATIC_DIR, "lime.png")

    plt.imsave(path, lime_img)
