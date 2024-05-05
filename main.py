import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import keras

model = keras.models.load_model('mnist.keras')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post('/predict-image/')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L').resize((28, 28), PIL.Image.ANTIALIAS)
    img_array = np.array(pil_image).reshape((1, 28, 28))
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)
    return {"digit": pred_label}