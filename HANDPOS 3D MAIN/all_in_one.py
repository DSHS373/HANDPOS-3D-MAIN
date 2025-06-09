import cv2 as cv
import mediapipe as mp
import numpy as np
import socket
import sys
from utils import DLT, get_projection_matrix
from read_arUco import Detecter
# utils.py는 calib.py에서 저장한 파라미터 파일들을 읽어서 사용합니다.

# UDP 소켓 설정 (예: 로컬호스트 127.0.0.1, 포트 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
serverAddressPort = (UDP_IP, UDP_PORT)

# 카메라 인덱스 (칼리브레이션 시 camera0는 0번, camera1은 내부적으로 인덱스 1번 사용)
cam0 = 0           # 카메라 1
cam1 = 2           # 카메라 2 (calib 파일에서는 인덱스 1번으로 되어 있음)

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = Detecter()

# 캘리브레이션 시 사용했던 해상도 (예: 640x480)
frame_shape = [480, 640]  # [height, width]

# 캘리브레이션된 카메라 파라미터로부터 각 카메라의 프로젝션 행렬 읽어오기
P0 = get_projection_matrix(cam0)      # camera1의 프로젝션 행렬
P1 = get_projection_matrix(cam1 - 1)    # camera2의 프로젝션 행렬 (calib 파일 인덱스 1)

def get_camera_center(P):
    """
    주어진 3x4 프로젝션 행렬 P에서 카메라 센터(3D 좌표)를 구합니다.
    P = [M | p4] 일 때, 카메라 센터는 -inv(M) @ p4 로 계산할 수 있습니다.
    """
    M = P[:, :3]
    p4 = P[:, 3]
    center = -np.linalg.inv(M) @ p4
    return center

def get_camera_rotation(P):
    """
    OpenCV의 decomposeProjectionMatrix를 사용하여 주어진 3x4 프로젝션 행렬 P에서
    회전 행렬(Rotation Matrix, 3x3)을 추출합니다.
    최신 OpenCV에서는 7개의 값을 반환합니다.
    """
    _, R, _, _, _, _, _ = cv.decomposeProjectionMatrix(P)
    return R

def run_handpose_udp(cam_idx0, cam_idx1, P0, P1):
    # 두 카메라 열기
    cap0 = cv.VideoCapture(cam_idx0, cv.CAP_DSHOW)
    cap1 = cv.VideoCapture(cam_idx1, cv.CAP_DSHOW)
    # 해상도 설정 (칼리브레이션 시 사용한 해상도)
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    
    # Mediapipe Hands 초기화 (한 프레임당 최대 1개의 손만 검출)
    hands0 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)

    # 센티미터 단위를 미터 단위로 변환하기 위한 스케일 팩터 (1 cm = 0.01 m) 1-> 10cm
    scale_factor = 0.1

    # 두 카메라의 센터와 회전 행렬 계산
    cam0_center = get_camera_center(P0)
    cam1_center = get_camera_center(P1)
    cam0_rotation = get_camera_rotation(P0)
    cam1_rotation = get_camera_rotation(P1)
    
    # 이미지 좌표계 기준 변환 (손 랜드마크와 동일한 방식 적용)
    cam0_center[0] -= (frame_shape[1] / 2)
    cam0_center[1] = (frame_shape[0] / 2) - cam0_center[1]
    cam1_center[0] -= (frame_shape[1] / 2)
    cam1_center[1] = (frame_shape[0] / 2) - cam1_center[1]
    # 스케일 팩터 적용 (미터 단위 변환)
    cam0_center = cam0_center * scale_factor
    cam1_center = cam1_center * scale_factor

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Error: 카메라에서 프레임을 받아올 수 없습니다.")
            break
        
        # BGR 이미지를 RGB로 변환 (Mediapipe는 RGB 이미지 사용)
        rgb0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        rgb1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        
        # 손 검출
        results0 = hands0.process(rgb0)
        results1 = hands1.process(rgb1)
        
        # 각 카메라에서 21개 손 랜드마크 (픽셀 좌표) 추출
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
            keypoints0 = [[-1, -1]] * 21  # 검출 실패 시
        
        if results1.multi_hand_landmarks:
            hand_landmarks = results1.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                x = int(round(lm.x * frame_shape[1]))
                y = int(round(lm.y * frame_shape[0]))
                keypoints1.append([x, y])
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            keypoints1 = [[-1, -1]] * 21
        
        # 두 카메라 모두에서 손이 검출되면 삼각측량 수행하여 3D 좌표 계산
        points_3d = []
        for uv0, uv1 in zip(keypoints0, keypoints1):
            if uv0[0] == -1 or uv1[0] == -1:
                points_3d.append([-1, -1, -1])
            else:
                # utils.py의 DLT 함수를 이용하여 3D 좌표 복원
                pt3d = DLT(P0, P1, uv0, uv1)
                # 이미지 중앙 기준 좌표 변환 (x: 오른쪽 양수, y: 위쪽 양수)
                pt3d[0] -= (frame_shape[1] / 2)
                pt3d[1] = (frame_shape[0] / 2) - pt3d[1]
                # 센티미터 단위를 미터 단위로 변환
                pt3d = [coord * scale_factor for coord in pt3d]
                points_3d.append(pt3d)
        
        # 손 랜드마크, 그리고 카메라 정보가 정상적으로 계산되었을 때만 데이터 전송
        if keypoints0[0][0] != -1 and keypoints1[0][0] != -1:
            # 1. 손 랜드마크 3D 좌표 (63개)
            flattened = [float(coord) for point in points_3d for coord in point]
            # 2. 카메라 1 센터 (3개)
            flattened.extend([float(c) for c in cam0_center])
            # 3. 카메라 1 회전 행렬 (9개) -> 행 우선 순서(flatten)
            flattened.extend([float(r) for r in cam0_rotation.flatten()])
            # 4. 카메라 2 센터 (3개)
            flattened.extend([float(c) for c in cam1_center])
            # 5. 카메라 2 회전 행렬 (9개) -> 행 우선 순서(flatten)
            flattened.extend([float(r) for r in cam1_rotation.flatten()])
            
            
            
            
            
        aruco1 = detector.detect(frame0 ,"center")
        aruco2 = detector.detect(frame1 ,"center")
        frame0 = detector.draw(frame0)
        frame1 = detector.draw(frame1)
        
        if (not aruco1 is None and not aruco2 is None and (len(aruco1) == len(aruco2))):
            for uv0, uv1 in zip(aruco1, aruco2):
                np.append(uv0, [0])
                np.append(uv1, [0])
                print(uv0, uv1)
                if uv0[0] == -1 or uv1[0] == -1:
                    points_3d.append([-1, -1, -1])
                else:
                    # utils.py의 DLT 함수를 이용하여 3D 좌표 복원
                    pt3d = DLT(P0, P1, uv0, uv1)
                    # 이미지 중앙 기준 좌표 변환 (x: 오른쪽 양수, y: 위쪽 양수)
                    pt3d[0] -= (frame_shape[1] / 2)
                    pt3d[1] = (frame_shape[0] / 2) - pt3d[1]
                    # 센티미터 단위를 미터 단위로 변환
                    pt3d = [coord * scale_factor for coord in pt3d]
                    flattened.extend([float(coord) for point in pt3d for coord in point])
                
        
        # 손 랜드마크, 그리고 카메라 정보가 정상적으로 계산되었을 때만 데이터 전송
        if keypoints0[0][0] != -1 and keypoints1[0][0] != -1:
            # 전송할 데이터: 63 + 3 + 9 + 3 + 9 = 87 float 값
            send_str = str(flattened)
            sock.sendto(send_str.encode(), serverAddressPort)
        
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
