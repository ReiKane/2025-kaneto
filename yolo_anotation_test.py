from ultralytics import YOLO

model = YOLO('yolo11n.pt')
#model = YOLO('yolo11l.pt')

results = model.train(data='datasets_2/data.yaml', epochs = 100, imgsz = 640)