import cv2 as cv
import mediapipe as mp
import numpy as np
import socket
import sys
from utils import DLT, get_projection_matrix
from read_arUco import Detecter

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "127.0.0.1"
UDP_PORT_HANDS = 5052
UDP_PORT_CAMERA = 5053
serverAddressPortHands = (UDP_IP, UDP_PORT_HANDS)
serverAddressPortCamera = (UDP_IP, UDP_PORT_CAMERA)

cam0 = 1
cam1 = 2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [480, 640]  # [height, width]

P0 = get_projection_matrix(cam0 - 1)
P1 = get_projection_matrix(cam1 - 1)

arUco_detector = Detecter()

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

    scale_factor = 0.1

    cam0_center = get_camera_center(P0)
    cam1_center = get_camera_center(P1)
    cam0_rotation = get_camera_rotation(P0)
    cam1_rotation = get_camera_rotation(P1)
    
    cam0_center[0] -= (frame_shape[1] / 2)
    cam0_center[1] = (frame_shape[0] / 2) - cam0_center[1]
    cam1_center[0] -= (frame_shape[1] / 2)
    cam1_center[1] = (frame_shape[0] / 2) - cam1_center[1]
    
    cam0_center = cam0_center * scale_factor
    cam1_center = cam1_center * scale_factor

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Error: 카메라에서 프레임을 받아올 수 없습니다.")
            break
        
        print(arUco_detector.detect(frame0, "center"))
        print(arUco_detector.detect(frame1, "center"))
        
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
        
        points_3d = []
        for uv0, uv1 in zip(keypoints0, keypoints1):
            if uv0[0] == -1 or uv1[0] == -1:
                points_3d.append([-1, -1, -1])
            else:
                pt3d = DLT(P0, P1, uv0, uv1)
                
                pt3d[0] -= (frame_shape[1] / 2)
                pt3d[1] = (frame_shape[0] / 2) - pt3d[1]
                
                pt3d = [coord * scale_factor for coord in pt3d]
                points_3d.append(pt3d)
        
        if True: # 카메라 UDP
            flattened_camera = []
            flattened_camera.extend([float(c) for c in cam0_center])
            flattened_camera.extend([float(r) for r in cam0_rotation.flatten()])
            flattened_camera.extend([float(c) for c in cam1_center])
            flattened_camera.extend([float(r) for r in cam1_rotation.flatten()])

            send_str_camera = str(flattened_camera)
            sock.sendto(send_str_camera.encode(), serverAddressPortCamera)
        
        if keypoints0[0][0] != -1 and keypoints1[0][0] != -1:
            flattened_hands = [float(coord) for point in points_3d for coord in point]

            send_str_hands = str(flattened_hands)
            sock.sendto(send_str_hands.encode(), serverAddressPortHands)
        
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