from flask import Flask, render_template, request
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Build and load the model
def load_custom_vgg_model():
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4608, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1152, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.load_weights('model_weights/vgg19_model_01.weights.h5')  # Ensure path and file are correct

    for layer in base_model.layers:
        layer.trainable = False

    return model

model = load_custom_vgg_model()
class_labels = ['No Tumor', 'Tumor']  # Update as per your classification labels

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(240, 240))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    return class_labels[class_index], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = model_predict(filepath, model)
            return render_template('index.html', result=label, confidence=round(confidence * 100, 2), image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
