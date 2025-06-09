import cv2
import cv2.aruco as aruco
import numpy as np
class Detecter:
    def __init__(self):
        # 정확한 딕셔너리 사용
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

        # 탐지 파라미터 조정
        self.parameters = aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.02
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.minDistanceToBorder = 2
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.01
        self.parameters.maxErroneousBitsInBorderRate = 0.04
        self.parameters.errorCorrectionRate = 0.6
        # 카메라 해상도 설정
        

        #  임시 카메라 내부 파라미터 (테스트용, 실제 보정값 아님)
        # 해상도 1280x720 기준, 중심은 (640, 360), 초점거리 1000 가정
        self.camera_matrix = np.array([[1000, 0, 640],
                                [0, 1000, 360],
                                [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1))  # 왜곡 없다고 가정
        self.marker_length = 0.08  # 마커 한 변 길이 (단위: m → 5cm)
    def detect(self,image,type):
        frame = image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            # 각 마커에 대해 pose 추정 → 좌표축 그리기
            #rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            if type=="corner"or type==0:
                return corners
            elif type=="center" or type==1:
                result=[]
                for i in range(len(ids)):
                    pts = corners[i][0]  # 4개의 꼭짓점
                    A = pts[0]
                    C = pts[2]
                    B = pts[1]
                    D = pts[3]
                    # 대각선 AC와 BD의 교차점 구하기 → 기하학적 중심
                    center = ((A + C) + (B + D)) / 4
                    result.append(center)
                return np.array(result)
            else:
                print("잘못된 type ㅇㅅㅇ")
    def draw(self,image):
        frame = image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 각 마커에 대해 pose 추정 → 좌표축 그리기
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids) 
                for i in range(len(ids)):       
                    for (x, y) in corners[i][0]:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)    
                        #cv2.putText( frame,f"({int(x)},{int(y)})", (int(x-20),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)
                    #print(f"ID {ids[i][0]} 위치: {tvecs[i].flatten()}")

        return frame
if __name__=="__main__":
    import cv2
    from datetime import datetime

    # 1. 클래스 인스턴스 생성
    d = Detecter()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()
    while True:
        # 2. 카메라에서 프레임 한 장 캡처
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 이미지를 읽을 수 없습니다.")
            exit()

        # 3. 이미지 저장 (예: 현재 시각 기준 파일명)
        image_path=frame
        print(d.detect(image_path,"center"))
        output = d.draw(image_path)

        # 5. 결과 이미지 보여주기
        if output is not None:
            cv2.imshow("Detected Markers with Axes", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()