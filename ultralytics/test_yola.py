from ultralytics import YOLO

# 加载best.pt模型
model = YOLO("/home/dell/ymd/yolo_max_yola/ultralytics/runs/detect/yola_exdark_20260312_085423/weights/best.pt")

# 验证模型在数据集上的表现
results = model.val(
    data="datasets/exdark.yaml",  # 替换成你的数据集yaml
    imgsz=960                     # 训练时的分辨率
)

# 打印关键指标
print("Precision:", results.metrics['P'])
print("Recall:", results.metrics['R'])
print("mAP50:", results.metrics['mAP50'])
print("mAP50-95:", results.metrics['mAP50-95'])