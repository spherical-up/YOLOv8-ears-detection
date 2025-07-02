from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
model.train(
    data='ear-data.yaml',   # 数据集配置文件
    epochs=50,              # 训练轮数，可根据需要调整
    imgsz=640,              # 输入图片尺寸
    batch=16,               # 批次大小，可根据显存调整
    project='runs/ear_yolo',# 训练结果保存目录
    name='exp',             # 实验名
)