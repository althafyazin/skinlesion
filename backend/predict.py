import numpy as np

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

def predict_image(model, img_tensor):
    preds = model.predict([img_tensor, img_tensor])[0]

    idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[idx]
    confidence = float(np.max(preds)) * 100

    probabilities = {
        CLASS_NAMES[i]: float(preds[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probabilities, idx
