from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # <-- Changed import
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- KEY CHANGE: Load the model ---
# We load the entire model file we saved from the notebook.
# This replaces the complex load_custom_vgg_model() function.
try:
    model = load_model('brain_tumor_model.h5')
    print("* Model loaded successfully")
except Exception as e:
    print(f"* Error loading model: {e}")
    print("* Please make sure 'brain_tumor_model.h5' is in the same directory as app.py")
    model = None

# --- KEY CHANGE: Updated Class Labels ---
# These match the 'no' and 'yes' folders from our generator
class_labels = ['No Tumor', 'Tumor'] 

# --- KEY CHANGE: Updated Prediction Function ---
def model_predict(img_path, model):
    # Load the image, resizing it to 224x224 (the size our model was trained on)
    img = load_img(img_path, target_size=(224, 224)) 
    
    # Convert to numpy array and rescale (0-1)
    img_array = img_to_array(img) / 255.0
    
    # Expand dimensions to create a "batch" of 1
    img_array = np.expand_dims(img_array, axis=0)

    # Get the prediction
    # Our 'sigmoid' model outputs a single value (probability)
    pred_prob = model.predict(img_array)[0][0]

    # Determine label and confidence based on the 0.5 threshold
    if pred_prob > 0.5:
        label = class_labels[1] # 'Tumor'
        confidence = pred_prob * 100
    else:
        label = class_labels[0] # 'No Tumor'
        confidence = (1 - pred_prob) * 100
        
    return label, confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file and model: # Check if model loaded successfully
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Get prediction
            label, confidence = model_predict(filepath, model)
            
            return render_template('index.html', 
                                   result=label, 
                                   confidence=round(confidence, 2), 
                                   image_path=filepath)
        elif not model:
            return render_template('index.html', error="Model not loaded. Please check server logs.")
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)