import os
import tensorflow as tf
import numpy as np
import cv2

# absolute path to backend/static
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

def make_gradcam_heatmap(img_tensor, model, last_conv_layer, pred_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_tensor, img_tensor])

        # predictions comes as list → extract tensor
        predictions = predictions[0]
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def save_gradcam(image, heatmap):
    os.makedirs(STATIC_DIR, exist_ok=True)
    path = os.path.join(STATIC_DIR, "gradcam.png")

    image = np.uint8(255 * image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(path, superimposed)
