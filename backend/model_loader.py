from keras.models import load_model

MODEL_PATH = "model/incepx_ensemble_best(new).keras"

def load_trained_model():
    model = load_model(MODEL_PATH)
    return model
