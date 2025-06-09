import cv2

# 카메라 0번과 1번 열기
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 카메라 열기 확인
if not cap1.isOpened():
    print("Error: Camera 0 could not be opened.")
    exit()
if not cap2.isOpened():
    print("Error: Camera 1 could not be opened.")
    exit()

while True:
    # 각 카메라에서 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Failed to capture frame(s).")
        break

    # 프레임 출력
    cv2.imshow('Camera 0', frame1)
    cv2.imshow('Camera 1', frame2)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
