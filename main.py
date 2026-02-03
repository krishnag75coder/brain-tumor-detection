import os
import gc
import numpy as np
from flask import Flask, render_template, request, send_from_directory

# Import TensorFlow stuff but DO NOT load the model globally
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, template_folder="templates")

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']


def predict_tumor(image_path):
    # 1. Load Model ONLY when needed (Lazy Loading)
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)

    try:
        # 2. Process Image
        print("Processing image...")
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Predict
        predictions = model.predict(img_array)
        idx = np.argmax(predictions)
        confidence = float(np.max(predictions))

        if class_labels[idx] == 'notumor':
            result_text = "No Tumor"
        else:
            result_text = f"Tumor: {class_labels[idx]}"

        return result_text, confidence

    except Exception as e:
        print(f"Error: {e}")
        return "Error processing image", 0.0

    finally:
        # 4. CRITICAL: DESTROY MODEL TO FREE RAM
        print("Cleaning up memory...")
        del model
        tf.keras.backend.clear_session()
        gc.collect()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # This will now take a few seconds longer, but won't crash
            result, confidence = predict_tumor(file_path)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=f"/uploads/{file.filename}"
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run()