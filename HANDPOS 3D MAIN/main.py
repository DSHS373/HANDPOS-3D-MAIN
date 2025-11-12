import cv2 as cv
import mediapipe as mp
import numpy as np
import socket
import json
from utils import DLT, get_projection_matrix_with_shift

# =========================
# UDP 설정
# =========================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT_HANDS = 5052
UDP_PORT_CAMERA = 5053
serverAddressPortHands = (UDP_IP, UDP_PORT_HANDS)
serverAddressPortCamera = (UDP_IP, UDP_PORT_CAMERA)

# =========================
# 실제 캡처 장치 인덱스 (환경에 맞게 조정)
# =========================
cam0 = 1
cam1 = 2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [480, 640]  # [height, width]

# ===== 새 월드 원점(기존 월드 좌표 기준) 설정 =====
# (보정판 특정 코너 등 "기존 월드" 좌표의 점을 입력하면 그 점이 새 원점(0,0,0)이 됩니다.)
custom_origin_w = np.array([0.0, 0.0, 0.0], dtype=float)

# ===== 월드 모드: True면 3D 좌표/카메라 위치 모두 "새 월드" 기준으로 일관 계산 =====
world_mode = True

# world_to_cameraX_rot_trans 파일을 사용하여 원점 이동 반영한 P 생성
# camera_id는 0,1,... 이므로 camX-1 사용
P0 = get_projection_matrix_with_shift(cam0 - 1, custom_origin_w,
                                      use_world_to_files=True,
                                      savefolder='camera_parameters/',
                                      prefix='world_to_')
P1 = get_projection_matrix_with_shift(cam1 - 1, custom_origin_w,
                                      use_world_to_files=True,
                                      savefolder='camera_parameters/',
                                      prefix='world_to_')

def decompose_RT_from_P(P):
    """
    OpenCV의 decomposeProjectionMatrix는 7개를 반환합니다.
    여기선 K, R, t_h(호모 좌표의 카메라 중심)만 사용합니다.
    """
    P = P.astype(np.float64)
    K, R, t_h, *_ = cv.decomposeProjectionMatrix(P)  # 7개 중 앞의 3개만 취함
    # t_h: 4x1 (Cx, Cy, Cz, W)^T  → 3D 카메라 중심 C = (x/w, y/w, z/w)
    C = (t_h[:3] / t_h[3]).reshape(3)
    return R, C

def rot_to_quat(R):
    """
    회전행렬(3x3) -> 쿼터니언 (w,x,y,z)
    """
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

# ---------- OpenCV -> Unity 좌표 변환 ----------
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

def run_handpose_udp(cam_idx0, cam_idx1, P0, P1):
    cap0 = cv.VideoCapture(cam_idx0, cv.CAP_DSHOW)
    cap1 = cv.VideoCapture(cam_idx1, cv.CAP_DSHOW)
    
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    
    hands0 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)

    # 보기 좋은 스케일 (3D 결과 & 카메라 위치에 동일 적용)
    scale_factor = 0.1

    # P에서 바로 R, C 추출 (새 원점 이동이 P에 반영돼 있으므로 C도 "새 월드" 기준, OpenCV 좌표계)
    R0_cv, C0_cv = decompose_RT_from_P(P0)
    R1_cv, C1_cv = decompose_RT_from_P(P1)

    # 카메라 위치를 Unity 좌표계로 변환
    C0_u = vec_cv_to_unity(C0_cv) * scale_factor
    C1_u = vec_cv_to_unity(C1_cv) * scale_factor

    # 회전도 Unity 좌표계로 변환 후 쿼터니언 계산
    R0_u = rot_cv_to_unity(R0_cv)
    R1_u = rot_cv_to_unity(R1_cv)
    q0_u = rot_to_quat(R0_u)
    q1_u = rot_to_quat(R1_u)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Error: 카메라에서 프레임을 받아올 수 없습니다.")
            break
        
        rgb0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        rgb1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        
        results0 = hands0.process(rgb0)
        results1 = hands1.process(rgb1)
        
        keypoints0 = []
        keypoints1 = []
        
        if results0.multi_hand_landmarks:
            hand_landmarks = results0.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                x = int(round(lm.x * frame_shape[1]))
                y = int(round(lm.y * frame_shape[0]))
                keypoints0.append([x, y])
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            keypoints0 = [[-1, -1]] * 21
        
        if results1.multi_hand_landmarks:
            hand_landmarks = results1.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                x = int(round(lm.x * frame_shape[1]))
                y = int(round(lm.y * frame_shape[0]))
                keypoints1.append([x, y])
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            keypoints1 = [[-1, -1]] * 21
        
        points_3d_unity = []
        for uv0, uv1 in zip(keypoints0, keypoints1):
            if uv0[0] == -1 or uv1[0] == -1:
                points_3d_unity.append([-1.0, -1.0, -1.0])
            else:
                # DLT 3D (OpenCV/현재 월드 좌표계)
                pt3d_cv = DLT(P0, P1, uv0, uv1)

                # world_mode=False인 경우에만 디버그용 픽셀 중심 보정(권장 X: 시각화용)
                if not world_mode:
                    pt3d_cv[0] -= (frame_shape[1] / 2.0)
                    pt3d_cv[1] = (frame_shape[0] / 2.0) - pt3d_cv[1]

                # ===== Unity 좌표로 변환: (x,y,z) -> (-x, z, -y) =====
                pt3d_u = vec_cv_to_unity(pt3d_cv)
                pt3d_u *= scale_factor
                points_3d_unity.append([float(pt3d_u[0]), float(pt3d_u[1]), float(pt3d_u[2])])
        
        # ===== 카메라 파라미터 UDP 송신 (JSON, 이미 Unity 좌표계로 변환 완료) =====
        camera_packet = {
            "world_mode": world_mode,
            "custom_origin_w": custom_origin_w.tolist(),  # 참고용(원점 정의)
            "scale": scale_factor,                        # 참고용(표시 스케일)
            "coord_system": "Unity",                      # 좌표계 명시
            "cam0": {
                "C": C0_u.tolist(),                       # 카메라 위치 (Unity)
                "R_flat": R0_u.flatten().tolist(),        # 회전행렬 flat (Unity)
                "quat_wxyz": q0_u.tolist()                # 쿼터니언 (w,x,y,z, Unity)
            },
            "cam1": {
                "C": C1_u.tolist(),
                "R_flat": R1_u.flatten().tolist(),
                "quat_wxyz": q1_u.tolist()
            }
        }
        sock.sendto(json.dumps(camera_packet).encode('utf-8'), serverAddressPortCamera)
        
        # ===== 손 키포인트(3D, Unity 좌표) UDP 송신 =====
        if keypoints0[0][0] != -1 and keypoints1[0][0] != -1:
            flattened_hands = [float(coord) for point in points_3d_unity for coord in point]
            sock.sendto(json.dumps({"points3d": flattened_hands}).encode('utf-8'), serverAddressPortHands)
        
        # 실시간 영상 출력
        cv.imshow("Camera 0", frame0)
        cv.imshow("Camera 1", frame1)
        
        # 'q' 키 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("프로그램 종료")
            break
    
    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_handpose_udp(cam0, cam1, P0, P1)
