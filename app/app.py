from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = tf.keras.models.load_model("models/digits_recognition_cnn.h5")

# Function to preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input
    return image

# API route for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    predicted_number = int(np.argmax(prediction))  # Get the highest probability digit
    
    return {"prediction": predicted_number}
