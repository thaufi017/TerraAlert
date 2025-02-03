from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/img'
app.config['OUTPUT_FOLDER'] = 'static/uploads'  # Folder to save processed images
app.config['MODEL_PATH'] = 'models/best_model.h5'  # Ensure this path is correct
app.config['ALLOWED_EXTENSIONS'] = {'h5'}

# Check if the model file exists
if not os.path.exists(app.config['MODEL_PATH']):
    raise FileNotFoundError(f"Model file not found at {app.config['MODEL_PATH']}. Please ensure the file exists.")

# Load the pre-trained model
model = tf.keras.models.load_model(app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('predict', filename=filename))
    return redirect(request.url)

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Load and preprocess the image
    with h5py.File(file_path, 'r') as hdf:
        data = np.array(hdf.get('img'))
        data[np.isnan(data)] = 0.000001

        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))

        input_data = np.zeros((1, 128, 128, 6))
        input_data[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED
        input_data[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN
        input_data[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE
        input_data[0, :, :, 3] = data_ndvi  # NDVI
        input_data[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE
        input_data[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION

    # Make prediction
    threshold = 0.5
    pred_img = model.predict(input_data)
    pred_img = (pred_img > threshold).astype(np.uint8)

    # Extract DEM and NDVI for the predicted landslide-prone areas
    dem = data[:, :, 13]  # DEM is typically stored in the 13th channel
    ndvi = data_ndvi  # NDVI is already calculated

    # Mask the DEM and NDVI using the predicted landslide-prone areas
    landslide_dem = dem[pred_img[0, :, :, 0] == 1]  # DEM values for landslide areas
    landslide_ndvi = ndvi[pred_img[0, :, :, 0] == 1]  # NDVI values for landslide areas

    # Calculate statistics for DEM and NDVI
    dem_mean = np.mean(landslide_dem) if landslide_dem.size > 0 else 0
    dem_std = np.std(landslide_dem) if landslide_dem.size > 0 else 0
    ndvi_mean = np.mean(landslide_ndvi) if landslide_ndvi.size > 0 else 0
    ndvi_std = np.std(landslide_ndvi) if landslide_ndvi.size > 0 else 0

    # Save processed images to the output folder
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    original_image = data[:, :, 3:0:-1]  # RGB image
    original_image_normalized = original_image / original_image.max()

    # Create overlay image (original image with mask)
    overlay_image = original_image_normalized.copy()
    overlay_image[pred_img[0, :, :, 0] == 1] = [1, 0, 0]  # Highlight mask in red

    # Save images
    original_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original.png')
    mask_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'mask.png')
    overlay_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'overlay.png')

    plt.imsave(original_image_path, original_image_normalized)
    plt.imsave(mask_image_path, pred_img[0, :, :, 0], cmap='gray')
    plt.imsave(overlay_image_path, overlay_image)

    # Save DEM and NDVI statistics to a CSV file
    stats = {
        'DEM Mean': [dem_mean],
        'DEM Std': [dem_std],
        'NDVI Mean': [ndvi_mean],
        'NDVI Std': [ndvi_std]
    }
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(app.config['OUTPUT_FOLDER'], 'landslide_stats.csv'), index=False)

    # Redirect to the results page with the image paths and statistics
    return redirect(url_for('results', 
                            original_image=original_image_path,
                            mask_image=mask_image_path,
                            overlay_image=overlay_image_path,
                            dem_mean=dem_mean,
                            dem_std=dem_std,
                            ndvi_mean=ndvi_mean,
                            ndvi_std=ndvi_std))

@app.route('/results')
def results():
    # Retrieve the image paths and statistics from the query parameters
    original_image = request.args.get('original_image')
    mask_image = request.args.get('mask_image')
    overlay_image = request.args.get('overlay_image')
    dem_mean = request.args.get('dem_mean')
    dem_std = request.args.get('dem_std')
    ndvi_mean = request.args.get('ndvi_mean')
    ndvi_std = request.args.get('ndvi_std')
    
    return render_template('results.html', 
                           original_image=original_image,
                           mask_image=mask_image,
                           overlay_image=overlay_image,
                           dem_mean=dem_mean,
                           dem_std=dem_std,
                           ndvi_mean=ndvi_mean,
                           ndvi_std=ndvi_std)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)