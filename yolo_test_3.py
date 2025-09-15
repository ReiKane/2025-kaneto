from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train4/weights/best.pt")
cap = cv2.VideoCapture("../piano_test_3.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# numpy配列に変換
frames = np.array(frames)

# 画素ごとに中央値を取って背景推定
background = np.median(frames, axis=0).astype(np.uint8)

cv2.imshow("Background", background)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("background.jpg", background)