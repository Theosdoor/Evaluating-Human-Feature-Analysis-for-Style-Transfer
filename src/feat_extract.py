import cv2
import numpy as np
from pathlib import Path

def extract_humans_from_video(model, video_path):
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

def score_detection(det):
    """Score based on multiple criteria"""
    score = 0
    
    # 1. Confidence (0-1)
    score += det['confidence'] * 0.3
    
    # 2. Size preference (want reasonably sized humans)
    # Penalize very small or very large (closeups/partial)
    img_area = det['frame'].shape[0] * det['frame'].shape[1]
    relative_area = det['area'] / img_area
    if 0.05 < relative_area < 0.4:  # reasonable human size
        score += 0.3
    
    # 3. Centering (humans near center often better framed)
    frame_h, frame_w = det['frame'].shape[:2]
    x1, y1, x2, y2 = det['bbox']
    center_x, center_y = (x1+x2)/2, (y1+y2)/2
    dist_from_center = np.sqrt((center_x - frame_w/2)**2 + (center_y - frame_h/2)**2)
    max_dist = np.sqrt((frame_w/2)**2 + (frame_h/2)**2)
    score += (1 - dist_from_center/max_dist) * 0.2
    
    # 4. Blur detection (avoid blurry frames)
    x1, y1, x2, y2 = det['bbox']
    patch = det['frame'][y1:y2, x1:x2]
    if patch.size > 0:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 100:  # threshold for "not blurry"
            score += 0.2
    
    return score

def diverse_sampling(detections, target_count=1000, temporal_gap=30):
    """Sample detections that are temporally spaced"""
    selected = []
    used_frames = set()
    
    for det in detections:
        # Check if too close to already selected frames
        too_close = any(abs(det['frame_num'] - f) < temporal_gap 
                       for f in used_frames)
        
        if not too_close and len(selected) < target_count:
            selected.append(det)
            used_frames.add(det['frame_num'])
    
    return selected

def save_patches(detections, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        patch = det['frame'][y1:y2, x1:x2]
        
        filename = f"human_{i:04d}_conf{det['confidence']:.2f}_score{det['score']:.2f}.jpg"
        cv2.imwrite(str(output_dir / filename), patch)

