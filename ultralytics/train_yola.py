from ultralytics import YOLO
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model = YOLO("ultralytics/models/yola_l.yaml")
model.load("yolov8l.pt")

print(model.info())

model.train(
    data="datasets/exdark.yaml",
    epochs=300,
    imgsz=960,
    batch=4,
    multi_scale=0.2,
    close_mosaic=10, 
    optimizer="AdamW",
    lr0=0.001,
    cache=True,
    workers=8,
    device=0,
    ii=5.0,
    ii_warmup=10,
    name=f"yola_exdark_{timestamp}"
)