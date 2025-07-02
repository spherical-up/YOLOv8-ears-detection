#!/usr/bin/env python3
"""
YOLOv8 耳朵识别模型测试脚本
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_model():
    """测试YOLOv8模型是否能正常加载和运行"""
    print("🔍 开始测试YOLOv8耳朵识别模型...")
    
    # 检查模型文件是否存在
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件 {model_path} 不存在")
        return False
    
    try:
        # 加载模型
        print("📦 正在加载模型...")
        import torch
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([
            DetectionModel,
            torch.nn.modules.container.Sequential
        ])
        model = YOLO(model_path)
        print("✅ 模型加载成功")
        
        # 检查是否有测试图片
        test_images = ['ear.jpg', 'face.webp']
        available_images = [img for img in test_images if os.path.exists(img)]
        
        if not available_images:
            print("⚠️  没有找到测试图片，创建一个简单的测试图像...")
            # 创建一个简单的测试图像
            test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.rectangle(test_img, (200, 150), (440, 330), (0, 0, 0), 2)
            cv2.imwrite('test_image.jpg', test_img)
            available_images = ['test_image.jpg']
        
        # 测试推理
        for img_path in available_images:
            print(f"🖼️  测试图片: {img_path}")
            
            # 进行预测，降低置信度阈值
            results = model(img_path, conf=0.1)
            
            # 显示结果
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                print(f"✅ 检测到 {len(boxes)} 个目标")
                for i, box in enumerate(boxes):
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    print(f"   目标 {i+1}: 类别 {class_id}, 置信度 {confidence:.3f}")
                    print(f"       边界框: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
            else:
                print("ℹ️  未检测到任何目标")
            
            print()
        
        print("🎉 模型测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_model() 