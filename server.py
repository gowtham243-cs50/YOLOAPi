from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import io
import uuid
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = os.environ.get("MODEL_PATH", "model/best.pt")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    # Will lazy-load on first request if there's an issue here

@app.get("/")
def read_root():
    return {"status": "active", "model": model_path}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process with YOLO
    try:
        # Load model if not loaded yet
        if 'model' not in globals():
            global model
            model = YOLO(model_path)
        
        # Run inference
        results = model(image)
        
        # Get image with annotations
        result_image = results[0].plot()
        
        # Convert to bytes for response
        img_byte_arr = io.BytesIO()
        Image.fromarray(result_image).save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        
        # Return image
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-base64")
async def predict_base64(file: UploadFile = File(...)):
    """Endpoint that returns base64 image - useful for some frontend frameworks"""
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process with YOLO
    try:
        # Run inference
        results = model(image)
        
        # Get image with annotations
        result_image = results[0].plot()
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        Image.fromarray(result_image).save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        # Return base64 string
        return {"image": img_base64, "content_type": "image/jpeg"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
