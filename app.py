import requests
import torch
import cv2
import numpy as np

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import StreamingResponse

from configs.api_configs import get_image_from_url
from process import process

# ================================================================= #

app = FastAPI()

@app.post("/glaucoma-classification")
async def glaucoma_classification(image_url: str = Query(..., description=".")):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        mine_type = response.headers.get('Content-Type', 'image/png')
        image = get_image_from_url(response.content)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve image from URL: {e}"}
    
    label, conf = process(image)

    results = {
        "label": label,
        "confidence": conf
    }

    return results

# @app.post("/glaucoma-classification")
# async def glaucoma_classification(file: UploadFile = File(...)):
#     try:
#         file_bytes = await file.read()
#         image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
#     except Exception as e:
#         return {"error": f"Failed to read file: {e}"}
    
#     label, conf = process(image)

#     return StreamingResponse(output, media_type="image/png")