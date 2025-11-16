# stereo_yolo_epipolar.py
import cv2
import numpy as np
from ultralytics import YOLO
from sort_tracker.sort import Sort

# ---------------------------
# Helper: extract 'box' detections (xyxy, cls, conf) and centers
# ---------------------------
def extract_box_detections(result, model, target_classes=None):
    """
    result : ultralytics result for a single image (results[0])
    model  : YOLO model (with .names)
    target_classes : (list[str] or None) → 특정 클래스만 필터링할 때 사용
    
    return : list of dicts
        {
            'xyxy': [x1,y1,x2,y2],
            'center': (cx, cy),
            'cls': cls_id,
            'name': cls_name,
            'conf': confidence
        }
    """
    dets = []
    if result is None or result.boxes is None:
        return dets
    
    boxes = result.boxes

    # numpy 변환 (GPU/CPU 상관없이 처리)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for bb, conf, cid in zip(xyxy, confs, cls_ids):
        name = model.names[cid] if hasattr(model, 'names') else str(cid)

        # 특정 클래스만 필터링 (예: ["person", "car"])
        if target_classes is not None and name not in target_classes:
            continue

        x1, y1, x2, y2 = bb.astype(int)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        dets.append({
            'xyxy': [x1, y1, x2, y2],
            'center': (cx, cy),
            'cls': cid,
            'name': name,
            'conf': float(conf)
        })
    return dets

# ---------------------------
# Main real-time loop
# ---------------------------


matchdict=dict()
last_len=0


def get_matchdict(frame1,frame2,model_path,tracker1,tracker2):
    global matchdict
    global last_len
    model = YOLO(model_path)
    conf_thresh = 0.8
    res1 = model.predict(frame1, imgsz=416, conf=conf_thresh, verbose=False)[0]
    dets1 = extract_box_detections(res1, model)
    res2 = model.predict(frame2, imgsz=416, conf=conf_thresh, verbose=False)[0]
    dets2 = extract_box_detections(res2, model)
    sort_input1 = []
    for d in dets1:
        x1, y1, x2, y2 = d["xyxy"]
        conf = d["conf"]
        sort_input1.append([x1, y1, x2, y2, conf])
    sort_input2 = []
    for d in dets2:
        x1, y1, x2, y2 = d["xyxy"]
        conf = d["conf"]
        sort_input2.append([x1, y1, x2, y2, conf])
    if len(sort_input1) > 0:
        sort_input1 = np.array(sort_input1, dtype=float)
    
    else:
        sort_input1 = np.empty((0, 5), dtype=float)
    tracked_objects1 = tracker1.update(sort_input1)

    if len(sort_input2) > 0:
        sort_input2 = np.array(sort_input2, dtype=float)
    else:
        sort_input2 = np.empty((0, 5), dtype=float)
    tracked_objects2 = tracker2.update(sort_input2)
    if len(tracked_objects1)==len(tracked_objects2) and len(tracked_objects1)!=last_len:
        matchdict=dict()
        to11=list(tracked_objects1)
        to1=sorted(tracked_objects1,key=lambda x:x[0])
        to2=sorted(tracked_objects2,key=lambda x:x[0])
        for i in range(len(to1)):
            obj1=to1[i]
            obj2=to2[i]
            track_id1=int(obj1[5])
            matchdict[track_id1]=obj2
            
        last_len=len(to1)
    
    resultdict=dict()
    id=0
    for obj1 in tracked_objects1:
        x1, y1, x2, y2 = obj1[:4]
        track_id=int(obj1[5])
        if track_id not in matchdict:
            continue
        resultdict[id]=[]
        resultdict[id].append([x1,y1,x2,y2])
        obj2=matchdict[track_id]
        x1, y1, x2, y2 = obj2[:4]
        resultdict[id].append([x1,y1,x2,y2])
        id+=1
    return resultdict


def main():
    cam_idx1 =1
    cam_idx2=2
    global matchdict
    global last_len
    asklsdfgksdj=0
    model_path = "models/best.pt"
    tracker1 = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
    tracker2 = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
    conf_thresh = 0.8
    cap1 = cv2.VideoCapture(cam_idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(cam_idx2, cv2.CAP_DSHOW)
    while True:
        ok1, frame1 = cap1.read()
        if not ok1 :
            break
        ok2, frame2 = cap2.read()
        if not ok2 :
            break
        print(get_matchdict(frame1,frame2,model_path,tracker1,tracker2))

if __name__ == "__main__":
    main()