import cv2 as cv
import mediapipe as mp
import numpy as np
import socket
import json
from utils import DLT, get_projection_matrix_with_shift
from ultralytics import YOLO
from sort_tracker.sort import Sort

# =========================
# UDP 설정
# =========================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT_HANDS = 5052
UDP_PORT_CAMERA = 5053
UDP_PORT_BOX = 5054
serverAddressPortHands = (UDP_IP, UDP_PORT_HANDS)
serverAddressPortCamera = (UDP_IP, UDP_PORT_CAMERA)
serverAddressPortBox = (UDP_IP, UDP_PORT_BOX)

# =========================
# 실제 캡처 장치 인덱스 (환경에 맞게 조정)
# =========================
cam0 = 1
cam1 = 2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [480, 640]  # [height, width]
H, W = frame_shape  # 편하게 쓰려고 캐시

# ===== 새 월드 원점(기존 월드 좌표 기준) 설정 =====
custom_origin_w = np.array([0.0, 0.0, 0.0], dtype=float)

# ===== 월드 모드: True면 3D 좌표/카메라 위치 모두 "새 월드" 기준 =====
world_mode = True

# ---- 프레임별 처리 주기 ----
DETECT_EVERY = 3   # YOLO + SORT 매칭은 3프레임마다 한 번
HAND_EVERY   = 2   # 손 Mediapipe는 2프레임마다 한 번

# world_to_cameraX_rot_trans 파일을 사용하여 원점 이동 반영한 P 생성
P0 = get_projection_matrix_with_shift(cam0 - 1, custom_origin_w,
                                      use_world_to_files=True,
                                      savefolder='camera_parameters/',
                                      prefix='world_to_')
P1 = get_projection_matrix_with_shift(cam1 - 1, custom_origin_w,
                                      use_world_to_files=True,
                                      savefolder='camera_parameters/',
                                      prefix='world_to_')

ANGLE_THES = 13  # 손-박스 각도 임계값 (deg)

################################################
# yolo 관련
################################################

def extract_box_detections(result, model, target_classes=None):
    dets = []
    
    if result is None or result.boxes is None:
        return dets
    
    boxes = result.boxes
    
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    
    for bb, conf, cid in zip(xyxy, confs, cls_ids):
        name = model.names[cid] if hasattr(model, 'names') else str(cid)
        
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

# 전역
matchdict = {}          # id1 -> id2
bbox_cache2 = {}        # id2 -> last bbox
ANGLE_THRESH = np.radians(5)   # polar 정렬 각도 버킷 크기

def should_recompute(tr1, tr2):
    # 두 카메라 모두 트랙이 있고, 수가 너무 다르지 않을 때만 재매칭
    return (len(tr1) > 0 and len(tr2) > 0 and abs(len(tr1) - len(tr2)) <= 1)

def _polar_key(o, angle_thresh=ANGLE_THRESH):
    """
    bbox 중심을 (0,0) 기준으로 polar 좌표(각도, 거리)로 변환하고
    각도는 angle_thresh 단위로 버킷팅해서,
    같은 버킷(=각도 비슷)끼리는 거리로 정렬되도록 key를 만든다.
    """
    x1, y1, x2, y2 = o[:4]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    angle = np.atan2(cy, cx)          # 라디안
    dist  = np.hypot(cx, cy)

    if angle_thresh > 0:
        angle_bucket = int(angle / angle_thresh)
    else:
        angle_bucket = 0

    # 먼저 각도 버킷(≈ 각도), 그다음 거리를 기준으로 정렬
    return (angle_bucket, dist)

def get_matchdict(frame1, frame2, model, tracker1, tracker2):
    """
    원래 코드 그대로: 여기서 YOLO + SORT + 매칭을 한 번에 처리.
    """
    global matchdict, bbox_cache2

    # 탐지
    results = model.predict([frame1, frame2], conf=0.9, verbose=False)
    res1, res2 = results[0], results[1]

    dets1 = extract_box_detections(res1, model)
    dets2 = extract_box_detections(res2, model)

    sort_input1 = np.array([[*d["xyxy"], d["conf"]] for d in dets1], dtype=float) if dets1 else np.empty((0,5), float)
    sort_input2 = np.array([[*d["xyxy"], d["conf"]] for d in dets2], dtype=float) if dets2 else np.empty((0,5), float)

    tracked_objects1 = tracker1.update(sort_input1)
    tracked_objects2 = tracker2.update(sort_input2)

    # 2번 카메라 bbox 캐시 갱신
    for o in tracked_objects2:
        bbox_cache2[int(o[5])] = o[:4].tolist()

    # 조건 충족시에만 매칭 재계산 (각도+거리 정렬 기반)
    if should_recompute(tracked_objects1, tracked_objects2):
        # 왼쪽 상단 (0,0) 기준 polar 정렬
        to1 = sorted(tracked_objects1, key=_polar_key)
        to2 = sorted(tracked_objects2, key=_polar_key)

        new_map = {}
        for a, b in zip(to1, to2):
            new_map[int(a[5])] = int(b[5])
        if new_map:                 # 유효할 때만 교체 (비는 프레임에서 비워지지 않게)
            matchdict = new_map

    # 현재 프레임의 bbox 조회 (없으면 캐시 사용)
    id2_now = {int(o[5]): o[:4].tolist() for o in tracked_objects2}

    resultdict = {}
    idx = 0
    for o1 in tracked_objects1:
        id1 = int(o1[5])
        if id1 not in matchdict:
            continue
        id2 = matchdict[id1]
        bbox1 = o1[:4].tolist()
        bbox2 = id2_now.get(id2, bbox_cache2.get(id2, None))
        if bbox2 is None:
            continue
        resultdict[idx] = [bbox1, bbox2]
        idx += 1

    return resultdict


def draw_yolo_matches(frame1, frame2, resultdict, selected):
    colors = [
        (0,0,255), (0,165,255), (0,255,255),
        (255,0,0), (255,0,255), (128,0,128), (42,42,165),
    ]
    green = (0, 255, 0)
    
    for idx, (obj_id, boxes) in enumerate(resultdict.items()):
        color = colors[idx % len(colors)]
        
        if idx in selected:
            color = green
        
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])
            cv.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame1, f'ID {obj_id}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        if len(boxes) > 1:
            x1, y1, x2, y2 = map(int, boxes[1])
            cv.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame2, f'ID {obj_id}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame1, frame2

################################################
# 투영/좌표계 유틸
################################################

def decompose_RT_from_P(P):
    P = P.astype(np.float64)
    K, R, t_h, *_ = cv.decomposeProjectionMatrix(P)
    C = (t_h[:3] / t_h[3]).reshape(3)
    return R, C

def rot_to_quat(R):
    q = np.empty(4, dtype=float)
    tr = np.trace(R)
    
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2,1] - R[1,2]) / s
        q[2] = (R[0,2] - R[2,0]) / s
        q[3] = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
    
    return q  # (w,x,y,z)

# OpenCV -> Unity 좌표 변환
M_cv2u = np.array([
    [ 0., -1.,  0.],
    [-1.,  0.,  0.],
    [ 0.,  0.,  1.]
], dtype=float)

def vec_cv_to_unity(v3):
    v3 = np.asarray(v3, dtype=float).reshape(3)
    return M_cv2u @ v3

def rot_cv_to_unity(R_cv):
    return M_cv2u @ R_cv @ M_cv2u.T

# ----- 박스 중심 3D 환원 → Unity 변환 → UDP(5054) 전송 -----

def format_points_packet(points_xyz):
    """
    points_xyz: [[x,y,z], ...] (Unity 좌표, float)
    포맷: [N,x1,y1,z1,x2,y2,z2,...]
    """
    flat = [len(points_xyz)]
    for p in points_xyz:
        flat.extend([float(p[0]), float(p[1]), float(p[2])])
    return "[" + ",".join(str(v) for v in flat) + "]"

def triangulate_yolo_centers_and_send(resultdict, P0, P1, scale_factor):
    """
    resultdict: {i: [ [x1,y1,x2,y2](cam0), [x1,y1,x2,y2](cam1) ] }
    각 페어의 bbox 중심을 u,v로 삼아 DLT → 3D (OpenCV) → Unity 변환 → UDP(5054) 전송
    """
    points_unity = []
    for _, pair in resultdict.items():
        if len(pair) < 2:
            continue
        # cam0 bbox 중심
        x10, y10, x20, y20 = pair[0]
        u0 = (x10 + x20) * 0.5
        v0 = (y10 + y20) * 0.5
        # cam1 bbox 중심
        x11, y11, x21, y21 = pair[1]
        u1 = (x11 + x21) * 0.5
        v1 = (y11 + y21) * 0.5

        pt3d_cv = DLT(P0, P1, [u0, v0], [u1, v1])

        if not world_mode:  # (옵션) 디버그용 픽셀 중심 보정
            pt3d_cv[0] -= (frame_shape[1] / 2.0)
            pt3d_cv[1] = (frame_shape[0] / 2.0) - pt3d_cv[1]

        pt3d_u = vec_cv_to_unity(pt3d_cv) * scale_factor
        points_unity.append([pt3d_u[0], pt3d_u[1], pt3d_u[2]])

    # 전송
    pkt = format_points_packet(points_unity)
    sock.sendto(pkt.encode('utf-8'), serverAddressPortBox)
    
    return points_unity

################################################

def angle_ABC(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    BA = A - B
    BC = C - B

    cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_theta))
    return theta

def main(cam_idx0, cam_idx1, P0, P1):
    cap0 = cv.VideoCapture(cam_idx0, cv.CAP_DSHOW)
    cap1 = cv.VideoCapture(cam_idx1, cv.CAP_DSHOW)
    
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, H)
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, H)
    
    hands0 = mp.solutions.hands.Hands(min_detection_confidence=0.5,
                                      max_num_hands=1,
                                      min_tracking_confidence=0.5)
    hands1 = mp.solutions.hands.Hands(min_detection_confidence=0.5,
                                      max_num_hands=1,
                                      min_tracking_confidence=0.5)
    
    draw_hands = False

    scale_factor = 0.1

    # 카메라 포즈 (OpenCV → Unity)
    R0_cv, C0_cv = decompose_RT_from_P(P0)
    R1_cv, C1_cv = decompose_RT_from_P(P1)
    C0_u = vec_cv_to_unity(C0_cv) * scale_factor
    C1_u = vec_cv_to_unity(C1_cv) * scale_factor
    R0_u = rot_cv_to_unity(R0_cv)
    R1_u = rot_cv_to_unity(R1_cv)
    q0_u = rot_to_quat(R0_u)
    q1_u = rot_to_quat(R1_u)
    
    # YOLO/SORT
    model_path = "models/best.pt"
    MODEL = YOLO(model_path)
    tracker1 = Sort(max_age=30, min_hits=3, iou_threshold=0.2)
    tracker2 = Sort(max_age=30, min_hits=3, iou_threshold=0.2)

    # Mediapipe 결과 / YOLO 결과 캐시
    keypoints0 = [[-1, -1]] * 21
    keypoints1 = [[-1, -1]] * 21
    points_3d_unity = [[-1.0, -1.0, -1.0]] * 21

    last_yolo_result_dict = {}
    last_box_poses = []

    # 카메라 파라미터 UDP: 카메라가 고정이면 한 번만 보내도 충분
    camera_packet = {
        "world_mode": world_mode,
        "custom_origin_w": custom_origin_w.tolist(),
        "scale": scale_factor,
        "coord_system": "Unity",
        "cam0": {
            "C": C0_u.tolist(),
            "R_flat": R0_u.flatten().tolist(),
            "quat_wxyz": q0_u.tolist()
        },
            "cam1": {
            "C": C1_u.tolist(),
            "R_flat": R1_u.flatten().tolist(),
            "quat_wxyz": q1_u.tolist()
        }
    }
    sock.sendto(json.dumps(camera_packet).encode('utf-8'), serverAddressPortCamera)

    frame_idx = 0
        
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Error: 카메라에서 프레임을 받아올 수 없습니다.")
            break

        frame_idx += 1

        # 첫 프레임은 무조건 실행, 이후부터는 주기적으로
        do_yolo  = (frame_idx == 1) or (frame_idx % DETECT_EVERY == 0)
        do_hands = (frame_idx == 1) or (frame_idx % HAND_EVERY   == 0)

        # =========================
        # YOLO: DETECT_EVERY 프레임마다만 get_matchdict 호출
        # =========================
        if do_yolo:
            yolo_result_dict = get_matchdict(frame0, frame1, MODEL, tracker1, tracker2)
            box_poses = triangulate_yolo_centers_and_send(yolo_result_dict, P0, P1, scale_factor)
            last_yolo_result_dict = yolo_result_dict
            last_box_poses = box_poses
        else:
            # YOLO 안 돌리는 프레임: 마지막 결과 재사용
            yolo_result_dict = last_yolo_result_dict
            box_poses = last_box_poses

        # =========================
        # Mediapipe Hands: HAND_EVERY 프레임마다만 예측 + UDP
        # =========================
        send_hands_udp = False

        if do_hands:
            rgb0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            rgb1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            results0 = hands0.process(rgb0)
            results1 = hands1.process(rgb1)

            keypoints0 = []
            keypoints1 = []

            if results0.multi_hand_landmarks:
                hand_landmarks0 = results0.multi_hand_landmarks[0]
                for lm in hand_landmarks0.landmark:
                    x = int(round(lm.x * W))
                    y = int(round(lm.y * H))
                    keypoints0.append([x, y])
                # 카메라 미리보기에서는 이 프레임에만 랜드마크 그려짐
                draw_hands = True
            else:
                keypoints0 = [[-1, -1]] * 21
                draw_hands = False
                
            if results1.multi_hand_landmarks:
                hand_landmarks1 = results1.multi_hand_landmarks[0]
                for lm in hand_landmarks1.landmark:
                    x = int(round(lm.x * W))
                    y = int(round(lm.y * H))
                    keypoints1.append([x, y])
                draw_hands = True
            else:
                keypoints1 = [[-1, -1]] * 21
                draw_hands = False

            # 손 3D 포인트 계산
            points_3d_unity = []
            for uv0, uv1 in zip(keypoints0, keypoints1):
                if uv0[0] == -1 or uv1[0] == -1:
                    points_3d_unity.append([-1.0, -1.0, -1.0])
                else:
                    pt3d_cv = DLT(P0, P1, uv0, uv1)
                    if not world_mode:
                        pt3d_cv[0] -= (W / 2.0)
                        pt3d_cv[1] = (H / 2.0) - pt3d_cv[1]
                    pt3d_u = vec_cv_to_unity(pt3d_cv) * scale_factor
                    points_3d_unity.append(
                        [float(pt3d_u[0]), float(pt3d_u[1]), float(pt3d_u[2])]
                    )

            # 이 프레임에만 UDP 전송 (예측 안 한 프레임은 안 보냄)
            send_hands_udp = True

        # =========================
        # 각도 계산 & 선택된 박스 찾아서 시각화
        # =========================
        angles = []
        selected = [(float("inf"), -1)]

        if len(points_3d_unity) > 8 and len(box_poses) > 0:
            pos5 = points_3d_unity[5]
            pos8 = points_3d_unity[8]

            for obj_id, pos in enumerate(box_poses):
                angle = angle_ABC(pos, pos5, pos8)
                angles.append((obj_id, angle))
                if angle < ANGLE_THES:
                    selected.append((angle, obj_id))

        # 선택된 박스 ID (없으면 -1)
        selected_id = min(selected)[1]
        frame0, frame1 = draw_yolo_matches(frame0, frame1, yolo_result_dict, [selected_id])
        
        if draw_hands:
            mp_drawing.draw_landmarks(frame0, hand_landmarks0, mp.solutions.hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame1, hand_landmarks1, mp.solutions.hands.HAND_CONNECTIONS)

        # =========================
        # 손 포인트 UDP (예측한 프레임에서만 전송)
        # =========================
        if send_hands_udp and keypoints0[0][0] != -1 and keypoints1[0][0] != -1:
            flattened_hands = [float(coord) for point in points_3d_unity for coord in point]
            sock.sendto(json.dumps({"points3d": flattened_hands}).encode('utf-8'), serverAddressPortHands)

        # 미리보기
        cv.imshow("Camera 0", frame0)
        cv.imshow("Camera 1", frame1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("프로그램 종료")
            break
    
    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(cam0, cam1, P0, P1)
