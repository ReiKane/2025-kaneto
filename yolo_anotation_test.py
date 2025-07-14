from ultralytics import YOLO

#model = YOLO('yolo11n.pt')
model = YOLO('yolo11l.pt')

results = model.train(data='datasets/data.yaml', epochs = 10, imgsz = 640)