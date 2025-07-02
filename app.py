import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import base64
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载YOLOv8模型
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from torch.nn.modules.conv import Conv2d

torch.serialization.add_safe_globals([
    DetectionModel,
    torch.nn.modules.container.Sequential,
    Conv,
    Conv2d
])
model = YOLO('best.pt')

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_ears(image_path):
    """使用YOLOv8检测耳朵"""
    try:
        # 使用模型进行预测，降低置信度阈值
        results = model(image_path, conf=0.1)
        
        # 获取检测结果
        result = results[0]
        boxes = result.boxes
        
        # 读取原始图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        detections = []
        
        if boxes is not None:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                print(f"检测到目标: 类别ID={class_id}, 置信度={confidence:.3f}, 框=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                
                # 绘制红色边界框
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                # 添加置信度标签
                label = f"Ear: {confidence:.2f}"
                draw.text((x1, y1-20), label, fill='red')
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id
                })
        
        # 转换回base64格式用于前端显示
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'success': True,
            'image': img_str,
            'detections': detections,
            'count': len(detections)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 进行耳朵检测
        result = detect_ears(filepath)
        
        # 删除上传的文件
        os.remove(filepath)
        
        return jsonify(result)
    
    return jsonify({'error': '不支持的文件格式'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8011) 