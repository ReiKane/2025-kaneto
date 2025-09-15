from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train4/weights/best.pt")
cap = cv2.VideoCapture("../piano_test_3.mp4")

xl = 10000
yu = 10000
xr = 0
yd = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_median = cv2.medianBlur(frame,5)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    sobel_x = cv2.Sobel(frame_median,cv2.CV_32F,1,0,ksize = 5)
    sobel_y = cv2.Sobel(frame_median,cv2.CV_32F,0,1,ksize = 5)
    sobel_x = cv2.convertScaleAbs(sobel_x, alpha = 0.5)
    sobel_y = cv2.convertScaleAbs(sobel_y, alpha = 0.5)
    sobel_xy = cv2.add(sobel_x, sobel_y)
    _, sobel_xy = cv2.threshold(sobel_xy, 150, 255, cv2.THRESH_BINARY)

    results = model.track(frame, classes=[1], max_det=7, persist=True)
    for frame_id, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
        xl = int(min(boxes[:, 0]))
        yu = int(min(boxes[:, 1]))
        xr = int(max(boxes[:, 2]))
        yd = int(max(boxes[:, 3]))
    sobel_xy = sobel_xy[yu:yd, xl:xr]

    length = 10
    gap = 10
    lines = cv2.HoughLinesP(sobel_xy.astype(np.uint8), 1, np.pi/180, 100, length, gap)
    sobel_xy = cv2.cvtColor(sobel_xy, cv2.COLOR_GRAY2BGR)
    print("lines: " + str(len(lines)))
    for x1,y1,x2,y2 in lines.squeeze():
        if(abs(x1-x2) <= 3):
            cv2.line(sobel_xy,(x1,y1),(x2,y2),(0,255,0),2)
    
    # フレームに結果を可視化
    annotated_frame = results[0].plot()
    cv2.imshow("src", annotated_frame)
    cv2.imshow("Sobel", sobel_xy)

    # qキーでプログラムを終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# キャプチャを終了
cap.release()
cv2.destroyAllWindows()