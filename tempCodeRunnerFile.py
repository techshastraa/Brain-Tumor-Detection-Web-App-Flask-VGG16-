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

# Build the model
def load_custom_vgg_model():
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)

    model = Model(base_model.input, output)

    # Load weights
    model.load_weights('model_weights/vgg19_model_01.weights.h5')


    # Freeze layers before block5_conv3
    set_trainable = False
    for layer in base_model.layers:
        if layer.name in ['block5_conv3', 'block5_conv4']:
            set_trainable = True
        layer.trainable = set_trainable

    return model

model = load_custom_vgg_model()
class_labels = ['Class 0', 'Class 1']  # Change according to your labels

# Prediction Function
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(240, 240))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    return class_labels[class_index], confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = model_predict(filepath, model)
            return render_template('index.html', result=label, confidence=round(confidence*100, 2), image_path=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
