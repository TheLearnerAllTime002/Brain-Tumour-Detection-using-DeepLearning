from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Uploads folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model, handle errors
try:
    model = load_model('models/model_v3.h5')
except Exception as e:
    print(f"Model loading error: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.lower().rsplit('.', 1)[1] in {'png','jpg','jpeg','bmp'}

# Prediction helper
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions[0]))
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    file_path = None
    error = None

    if request.method == 'POST':
        # Check for model
        if model is None:
            error = "Model failed to load. Please contact admin."
            return render_template('index.html', error=error, result=None)

        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            error = "Invalid or missing file. Please upload a valid image."
            return render_template('index.html', error=error, result=None)
        try:
            filename = file.filename
            safe_filename = os.path.basename(filename)
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(file_location)

            # Predict result
            result, confidence = predict_tumor(file_location)
            file_path = f'/uploads/{safe_filename}'
            return render_template('index.html', result=result,
                            confidence=f"{confidence*100:.2f}%", file_path=file_path)
        except Exception as e:
            error = f"Prediction error: {e}"
            return render_template('index.html', error=error, result=None)

    return render_template('index.html', error=None, result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
