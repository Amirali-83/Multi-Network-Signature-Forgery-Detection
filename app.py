from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os
from keras.layers import TFSMLayer

# Initialize Flask app
app = Flask(__name__)

MODEL_PATH = "signature_model_tf"

# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model folder '{MODEL_PATH}' not found. Place it in the project directory.")

# ✅ Load SavedModel using TFSMLayer (Keras 3 style)
model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
print("✅ Model loaded successfully using TFSMLayer!")

# Image size (same as used during training in Kaggle)
IMG_SIZE = (155, 220)

def preprocess_image(image_file):
    """
    Preprocess uploaded image: convert to grayscale, resize, normalize.
    """
    img = Image.open(image_file).convert("L").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel
    return np.expand_dims(img_array, axis=0)  # Add batch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']

        img1 = preprocess_image(file1)
        img2 = preprocess_image(file2)

        # ✅ With TFSMLayer, we call the model directly
        score = model([img1, img2])[0][0]

        return render_template('result.html', score=score)
    except Exception as e:
        return f"Error processing request: {e}"

if __name__ == '__main__':
    app.run(debug=True)
