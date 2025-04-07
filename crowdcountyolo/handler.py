from ultralytics import YOLO
import cv2
import requests
import pickle
import json
import base64
import numpy as np

def handle(req):
    try:
        # 1. Parse input data
        data = json.loads(req)
        img_data = base64.b64decode(data["image_data"]["image"])
        img = pickle.loads(img_data)
        
        # 2. Run detection
        model = YOLO("yolo11n.pt")
        results = model(img, classes=[0], conf=0.5, verbose=False)
        
        # 3. Prepare response data
        detections = []
        for result in results:
            for box in result.boxes:
                # Get detection info
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = box.conf[0].item()     # Confidence
                
                detections.append({
                    "bbox": xyxy,
                    "confidence": conf
                })
        
        # 4. Optionally encode processed image
        plotted_img = results[0].plot()  # Image with boxes drawn
        _, buffer = cv2.imencode('.jpg', plotted_img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        # 5. Return both detections and processed image
        return json.dumps({
            "detections": detections,
            "processed_image": encoded_img,
            "count": len(detections),
            "status": "success"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })
