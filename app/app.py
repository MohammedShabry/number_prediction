from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the pre-trained model (make sure the model file is in the correct path)
model = tf.keras.models.load_model('models/digits_recognition_cnn.h5')


# Preprocessing function to convert image data to the right format for the model
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 28, 28, 1)  # Reshape to match input shape
    return image

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    image = preprocess_image(image_bytes)
    
    prediction = model.predict(image)
    predicted_number = int(np.argmax(prediction))  # Get the digit with the highest probability
    
    return jsonify({'prediction': predicted_number})

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

