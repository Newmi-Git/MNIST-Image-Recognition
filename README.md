# MNIST-Image-Recognition

A neural network trained on the MNIST dataset, served through a live web interface — draw a digit in the browser and get a real-time prediction.

![demo](demo.gif)

## How it works

- A simple feedforward neural network (`model.py`) is trained on the MNIST dataset (`mnist_train.py`) and saved to `model_weights.pth`.
- A FastAPI backend (`app.py`) loads the trained model and exposes a `/predict` endpoint.
- The frontend (`server/`) is a canvas the user draws on. On each stroke, the drawing is sent to `/predict`, preprocessed to match the model's training data (cropped, centered, resized to 28x28), and the prediction is shown live.

## Running it locally

1. Clone the repo and install dependencies:

pip install -r requirements.txt

2. Train the model (this generates `model_weights.pth`):

python mnist_train.py

3. Start the server:

uvicorn app:app --reload

4. Open `http://127.0.0.1:8000` and draw a number.

## Tech stack

- **Model**: PyTorch
- **Backend**: FastAPI
- **Frontend**: HTML / CSS / vanilla JavaScript

## Notes

- The model is trained on the standard MNIST dataset (28x28, centered digits), so the preprocessing step crops and centers whatever the user draws before it reaches the model — this matters more than people expect for accuracy on freehand input.

Swap demo.gif for an actual short clip or screenshot once you have one — or delete that line if you don't want to bother making a GIF.