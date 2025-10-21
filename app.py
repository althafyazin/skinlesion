import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------------
# Load your trained model
# -----------------------------
model_path = 'model\incepx_ensemble_best.h5'
model = load_model(model_path)

# Mapping short names to full names
class_names = {
    'akiec': 'Actinic Keratosis and Intraepithelial Carcinoma',
    'bcc': 'Basal-Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Non-venereal Dermatoses',
    'vasc': 'Vasculitis'
}

# Get list of short names in the same order as your model outputs
short_names = list(class_names.keys())

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        if not file:
            return redirect(request.url)
        
        # Save file to uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess image for model
        IMG_SIZE = (299, 299)  # same as model input
        img = Image.open(filepath).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        
        # Predict
        preds = model.predict([img_array,img_array])
        pred_index = np.argmax(preds[0])
        pred_class_short = short_names[pred_index]
        pred_class_full = class_names[pred_class_short]
        
        return render_template('index.html', prediction=pred_class_full, image_url=filepath)
    
    return render_template('index.html')

# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
