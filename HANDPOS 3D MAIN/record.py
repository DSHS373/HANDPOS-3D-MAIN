print("Starting process. Please wait.")

import cv2
import sys

def initialize_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def get_fourcc():
    return cv2.VideoWriter_fourcc(*'mp4v')

try:
    camera_count = int(input("Enter how many cameras you want to use. The input must be a single integer (ex: 1, 2, 3, ...)\n>> "))
    if camera_count < 1:
        raise ValueError("The number of cameras must be at least 1.")
except Exception as e:
    print(e)
    print("Error: Wrong input. The input must be a single integer (ex: 1, 2, 3, ...)")
    sys.exit()

cameras = []

for i in range(camera_count):
    print(f"Camera {i} initializing...")
    cap = initialize_camera(i)
    if cap is None:
        print(f"Error: Camera {i} could not be initialized.")
        sys.exit()
    cameras.append(cap)
    print(f"Camera {i} ready")

recording = False
video_writers = [None] * camera_count

while True:
    frames = []
    for idx, cap in enumerate(cameras):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read from camera {idx}")
            sys.exit()
        
        if idx == 0 and not recording:
            cv2.putText(frame, "Press Space to start recording", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        
        frames.append(frame)
    
    for idx, frame in enumerate(frames):
        window_name = f"Camera {idx}"
        cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if not recording:
            print("Recording started.")
            for idx, cap in enumerate(cameras):
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                filename = f"media/camera{idx}_output.mp4"
                video_writers[idx] = cv2.VideoWriter(filename, get_fourcc(), 20.0, (frame_width, frame_height))
                if not video_writers[idx].isOpened():
                    print(f"Error: Could not open VideoWriter for camera {idx}")
                    sys.exit()
            recording = True
        else:
            print("Recording stopped.")
            for idx, writer in enumerate(video_writers):
                if writer is not None:
                    writer.release()
                    print(f"Camera {idx} video saved.")
                    video_writers[idx] = None
            recording = False
            break
    
    if recording:
        for idx, writer in enumerate(video_writers):
            if writer is not None:
                writer.write(frames[idx])
    
    if key == ord('q'):
        print("Exiting program.")
        break

for idx, cap in enumerate(cameras):
    cap.release()
    print(f"Camera {idx} released.")

for idx, writer in enumerate(video_writers):
    if writer is not None:
        writer.release()
        print(f"Camera {idx} VideoWriter released.")

cv2.destroyAllWindows()
