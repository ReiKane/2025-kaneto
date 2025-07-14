from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(data='datasets/data.yaml', epochs = 5, imgsz = 640)