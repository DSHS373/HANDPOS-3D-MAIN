#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import mediapipe as mp
import socket
from utils import DLT, get_projection_matrix
from read_arUco import Detecter

# UDP 통신 설정
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = (UDP_IP, UDP_PORT)

def initialize_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

if __name__ == '__main__':
    # 카메라 인덱스 입력
    try:
        # cams = input("Enter two camera indices separated by space (ex: '0 1'):\n>> ").split()
        cams = [0, 1]
        if len(cams) != 2:
            raise ValueError("Exactly two indices required.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 카메라 초기화
    cameras = []
    for idx in cams:
        print(f"Initializing camera {idx}...")
        cap = initialize_camera(idx)
        if cap is None:
            sys.exit(1)
        cameras.append(cap)
    print("Both cameras ready.")

    # 투영 행렬 로드
    P0 = get_projection_matrix(cams[0])
    P1 = get_projection_matrix(cams[1])

    # MediaPipe Hands 설정
    mp_hands = mp.solutions.hands
    hands0 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5,
                            max_num_hands=1,
                            min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # ArUco 디텍터 초기화
    # detector = Detecter()

    frame_shape = [480, 640]  # [height, width]
    scale_factor = 0.1        # 1cm -> 0.1m

    frame_idx = 0
    while True:
        # 프레임 획득
        frames = []
        for cap in cameras:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                sys.exit(1)
            frames.append(frame)

        # 손 검출을 위해 BGR -> RGB 변환
        rgb0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        rgb1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)

        # 처리 성능을 위해 쓰기 불가로 설정
        rgb0.flags.writeable = False
        rgb1.flags.writeable = False

        results0 = hands0.process(rgb0)
        results1 = hands1.process(rgb1)

        # 2D 손 랜드마크 추출
        pts0 = []
        pts1 = []
        h, w = frame_shape
        # print(h,w)
        if results0.multi_hand_landmarks:
            lm0 = results0.multi_hand_landmarks[0]
            for lm in lm0.landmark:
                print((lm.x * w) - w/2, (lm.y * h) * -1 + h/2)
                pts0.append([(lm.x * w) - w/2, (lm.y * h) * -1 + h/2])
            mp_drawing.draw_landmarks(frames[0], lm0, mp_hands.HAND_CONNECTIONS)
        else:
            pts0 = [[-1, -1]] * 21

        if results1.multi_hand_landmarks:
            lm1 = results1.multi_hand_landmarks[0]
            for lm in lm1.landmark:
                pts1.append([(lm.x * w) - w/2, (lm.y * h) * -1 + h/2])
            mp_drawing.draw_landmarks(frames[1], lm1, mp_hands.HAND_CONNECTIONS)
        else:
            pts1 = [[-1, -1]] * 21

        # 3D 손 좌표 삼각측량
        hand_3d = []
        for uv0, uv1 in zip(pts0, pts1):
            if uv0[0] == -1 or uv1[0] == -1:
                hand_3d.append([-1, -1, -1])
            else:
                p3d = DLT(P0, P1, uv0, uv1)
                hand_3d.append([coord * scale_factor for coord in p3d])

        # # ArUco 검출 및 3D 위치 계산 (선택)
        # ar0 = detector.detect(frames[0], "center")
        # ar1 = detector.detect(frames[1], "center")
        # ar_3d = []
        # if ar0 is not None and ar1 is not None and len(ar0) == len(ar1):
        #     for a0, a1 in zip(ar0, ar1):
        #         if a0[0] == -1 or a1[0] == -1:
        #             ar_3d.append([-1, -1, -1])
        #         else:
        #             p3d = DLT(P0, P1, a0, a1)
        #             ar_3d.append([coord * scale_factor for coord in p3d])
        #         detector.draw(frames[0])
        #         detector.draw(frames[1])

        # UDP 전송 데이터 구성 (hand_3d + ar_3d)
        # data = []
        # for pt in hand_3d:
        #     data.extend([float(c) for c in pt])
        # for pt in ar_3d:
        #     data.extend([float(c) for c in pt])
        # print(data)
        # 보내기
        flattened = [float(coord) for point in hand_3d for coord in point]
        # msg = np.array(data, dtype=np.float32).tobytes()
        # sock.sendto(msg, server_address)
        send_str = str(flattened)
        sock.sendto(send_str.encode(), server_address)

        # 디버깅용 출력
        # print(f"Frame {frame_idx}: Sent {len(data)} coordinates")
        frame_idx += 1

        # 화면 표시
        cv2.imshow(f"Cam {cams[0]}", frames[0])
        cv2.imshow(f"Cam {cams[1]}", frames[1])

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    # 자원 해제
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
