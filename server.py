from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("best.pt")  # Load trained YOLOv8 model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model(image)
    return {"detections": results[0].boxes.data.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
