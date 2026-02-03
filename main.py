import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

# We use the lighter 'Interpreter' instead of the full Keras load_model
import tensorflow.lite as tflite

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: We are now looking for the .tflite file
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']


def predict_tumor(image_path):
    try:
        # 1. Load the TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        # 2. Get input and output details.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 3. Preprocess the image manually (Keras helpers are heavy)
        # Resize to 128x128 (matches your training)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32)

        # Normalize (Assuming your model was trained with / 255.0)
        img_array = img_array / 255.0

        # Add batch dimension (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # 5. Run the inference
        interpreter.invoke()

        # 6. Get the result
        predictions = interpreter.get_tensor(output_details[0]['index'])

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