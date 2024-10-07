from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["cat.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen

#
# app = FastAPI()
#
#
# @app.post("/detect")
# async def detect_uav(file: UploadFile = File(...)):
#     # Получение видео/изображения и обработка с помощью YOLO11
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))
#
#     # Прогон через модель YOLO11
#     results = model(image)
#
#     # Возврат результатов
#     return {"detections": results}
