import cv2
import numpy as np
from pathlib import Path

def extract_humans_from_video(model, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO every N frames to save time
        if frame_count % 10 == 0:  # adjust based on video length
            results = model(frame, classes=[0])  # only detect persons
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf.item()
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Store detection info
                    detections.append({
                        'frame_num': frame_count,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'frame': frame.copy(),
                        'area': (x2-x1) * (y2-y1)
                    })
        
        frame_count += 1
    
    cap.release()
    return detections