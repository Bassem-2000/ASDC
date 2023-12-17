from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import cv2
import numpy as np
from face_attendance_json import classify_image

app = FastAPI()

@app.post("/attendance")
async def face_attendance(token:str,asdc_client_service_id:int ,file: UploadFile = File(...)):
    if not file.content_type.startswith('image'):
        return {"error": "Please upload an image file."}
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    name, QRcode, img_url = classify_image(img_np, token, asdc_client_service_id)

    return {"name": name,"QRcode":QRcode, "img":img_url}