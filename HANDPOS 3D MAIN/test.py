import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 설정 (옵션)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# 확인용 출력
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {int(width)} x {int(height)}")

# FPS 측정 변수
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame_count += 1
    elapsed_time = time.time() - start_time

    # 매 1초마다 FPS 출력
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"Measured FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
