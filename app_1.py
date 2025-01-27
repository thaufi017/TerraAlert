from flask import Flask, render_template, request, url_for
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import h5py

app = Flask(__name__)

# Load pre-trained model
try:
    model = tf.keras.models.load_model("best_model.h5")  # Update the path as needed
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file is a valid .h5 file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'h5'

def load_h5_data(h5_file_path):
    """Load and preprocess .h5 data."""
    with h5py.File(h5_file_path, 'r') as f:
        if 'img' not in f:
            raise ValueError("Key 'img' not found in the .h5 file.")
        data = f['img'][:]
        if data.shape != (128, 128, 14):
            raise ValueError(f"Unexpected data shape: {data.shape}.")
        # Preprocessing: Normalize and select the required channels
        data = data[:, :, :6] / 255.0  # Example preprocessing
        return data

def visualize_prediction(prediction, output_image_path):
    """Generate and save a visualization of the prediction in binary format."""
    plt.figure(figsize=(6, 6))
    plt.imshow(prediction, cmap='viridis', interpolation='none')  # Use categorical colormap
    plt.title("Predictions")
    plt.colorbar(label="Prediction Value")  # Optional: Include a colorbar for clarity
    plt.axis('on')  # Show axes for reference
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Load and preprocess data
            data = load_h5_data(filepath)
            input_data = np.expand_dims(data, axis=0)

            # Make prediction
            prediction = model.predict(input_data)[0]  # Get the first batch output

            # Normalize prediction for visualization
            prediction = prediction.squeeze()  # Ensure 2D
            threshold = 0.5  # Binary threshold
            prediction_binary = (prediction > threshold).astype(np.uint8)

            # Save the visualization and binary mask
            output_image_path = os.path.join(OUTPUT_FOLDER, "output.png")
            visualize_prediction(prediction, output_image_path)

            binary_output_path = os.path.join(OUTPUT_FOLDER, "output_binary.png")
            plt.imsave(binary_output_path, prediction_binary, cmap='gray')

            return render_template('result.html', 
                                   image_url=url_for('static', filename='outputs/output.png'),
                                   binary_url=url_for('static', filename='outputs/output_binary.png'))
        except Exception as e:
            return f"Error: {e}"
    return "Invalid file type. Upload a valid .h5 file."

if __name__ == '__main__':
    app.run(debug=True)
