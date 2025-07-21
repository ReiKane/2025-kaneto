from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")
cap = cv2.VideoCapture("../piano_test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # 物体検出（動画ではtrackメソッドを利用する）
    results = model.track(frame, iou=0.2, classes=[1], max_det=7, persist=True)
    # 物体検出（静止画ではdetectメソッドを利用する）
    # results = model.detect("path_to_image.jpg")
    
    # フレームに結果を可視化
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11トラッキング", annotated_frame)

    # qキーでプログラムを終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# キャプチャを終了
cap.release()
cv2.destroyAllWindows()