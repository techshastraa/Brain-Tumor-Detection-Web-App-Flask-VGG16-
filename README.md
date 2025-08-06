# Brain-Tumor-Detection-Web-App-Flask-VGG16-
This is a web-based application that allows users to upload brain MRI images and predicts whether the image indicates a brain tumor or no tumor using a fine-tuned VGG16 Convolutional Neural Network model.

brain-tumor-detection/
│
├── static/
│   └── styles.css              # CSS styling for frontend
│
├── templates/
│   ├── index.html              # Main page with upload form
│   └── result.html             # Result page showing prediction
│
├── model/
│   └── vgg16_model.h5          # Trained VGG16 model
│
├── app.py                      # Main Flask app
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


Requirements
Install the following dependencies:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt

txt
Copy
Edit
Flask==2.3.3
tensorflow==2.13.0
numpy
Pillow
🚀 How to Run
Clone the repository or download the project.

Move the trained model (vgg16_model.h5) to a folder named model/.

Start the Flask server:

bash
Copy
Edit
python app.py
Open your browser and navigate to:

cpp
Copy
Edit
http://127.0.0.1:5000/
📷 Usage
Upload a brain MRI image in .jpg, .jpeg, or .png format.

Click "Predict".

View the prediction result on the result page.

🔍 Model Details
Model: Transfer Learning using VGG16 (pretrained on ImageNet)

Input Size: 224x224 pixels

Output: Binary Classification – "Tumor" or "No Tumor"

📌 Notes
The frontend uses basic HTML and CSS for demonstration.

The app resizes and preprocesses the uploaded image before prediction.

📧 Contact
For improvements, issues, or feedback, feel free to connect.
adarshuniyal2003@gmail.com
