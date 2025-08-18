from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train4/weights/best.pt")
cap = cv2.VideoCapture("../piano_test.mp4")

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
    _, sobel_xy = cv2.threshold(sobel_xy, 182, 255, cv2.THRESH_BINARY)

    length = 100
    gap = 5
    lines = cv2.HoughLinesP(sobel_xy.astype(np.uint8), 1, np.pi/180, 100, length, gap)
    for x1,y1,x2,y2 in lines.squeeze():
        if(np.sqrt((x2-x1)**2 + (y2-y1)**2) < 100):
            cv2.line(sobel_xy,(x1,y1),(x2,y2),(0,255,0),2)
    
    results = model.track(frame, classes=[1], max_det=7, persist=True)
    
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