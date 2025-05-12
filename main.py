from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import logging
import os
import requests

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Enable CORS (edit allowed origins for production use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global settings
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 2  # Expected input channels
model = None

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "model.h5")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        logger.info("model.h5 not found. Downloading from Google Drive…")
        url = "https://drive.google.com/uc?export=download&id=1_WF39WHIFSSQOIWNKCNLMzuk68ofmM9G"
        resp = requests.get(url)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(resp.content)
        logger.info("Downloaded model.h5 successfully.")

    # load the model as before
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded. Input shape: {model.input_shape}")



# Image preprocessing function
def preprocess_image(image: Image.Image):
    try:
        image = image.convert("L")  # Grayscale
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image).astype(np.float32) / 255.0

        # Expand dimensions: (128, 128) -> (128, 128, 1)
        image_array = np.expand_dims(image_array, axis=-1)

        # Add batch dimension: (128, 128, 1) -> (1, 128, 128, 1)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please try again later."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = preprocess_image(image)

        logger.info(f"Preprocessed image shape: {image_array.shape}")
        expected_shape = model.input_shape[1:]
        actual_shape = image_array.shape[1:]

        if expected_shape != actual_shape:
            logger.warning(f"Input shape mismatch! Model expects {expected_shape}, got {actual_shape}")

        predictions = model.predict(image_array)
        prediction = predictions[0]

        classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        predicted_index = int(np.argmax(prediction))
        predicted_class = classes[predicted_index]
        confidence = float(prediction[predicted_index])

        all_predictions = dict(zip(classes, map(float, prediction)))

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

# Run the app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8220, reload=True)
