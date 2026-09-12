import base64
import io
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from model import ImageRecog


app = FastAPI()

model = ImageRecog()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

class ImagePayload(BaseModel):
    image: str

def preprocess(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28,28))
    #normalizing the image values
    arr = np.array(image, dtype=np.float32) / 255.
    tensor = torch.tensor(arr).view(1, -1)
    return tensor

@app.post("/predict")
def predict(payload:ImagePayload):
    encoded = payload.image.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(encoded)))

    tensor = preprocess(img)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    return {
        "prediction": int(probs.argmax()),
        "probabilities": probs.tolist()
    }

app.mount("/static", StaticFiles(directory="server"))