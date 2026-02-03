import os
import gc  # <--- IMPORT THIS
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# ---------- FIXED MODEL PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Load model globally
model = load_model(MODEL_PATH, compile=False)
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_tumor(image_path):
    try:
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        idx = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Result logic
        if class_labels[idx] == 'notumor':
            result_text = "No Tumor"
        else:
            result_text = f"Tumor: {class_labels[idx]}"

        return result_text, confidence

    except Exception as e:
        return str(e), 0.0

    finally:
        # <--- CRITICAL: CLEAN UP MEMORY --->
        gc.collect()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

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