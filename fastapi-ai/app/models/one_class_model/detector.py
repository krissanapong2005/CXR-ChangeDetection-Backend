# app/models/one_class_model/detector.py
import torch
import cv2
import numpy as np
import base64
import os
from .architecture import Autoencoder

class AnomalyDetector:
    def __init__(self, model_path, threshold=None):
        # 1. ตั้งค่าพื้นฐาน
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold if threshold else 0.001949
        
        # 2. โหลดโครงสร้างโมเดล
        self.model = Autoencoder().to(self.device)
        
        # 3. โหลด Weight (ใช้ path ที่รับเข้ามาโดยตรง)
        try:
            # map_location เพื่อให้รันได้ทั้งบนเครื่องที่มี/ไม่มี GPU
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"AnomalyDetector loaded weights from: {model_path}")
        except Exception as e:
            print(f"Error loading AnomalyDetector weights: {e}")
            raise e

    def preprocess(self, img_array):
        """
        Input: Numpy Array (ภาพที่ Decode มาแล้ว)
        Output: Tensor (1, 1, 256, 256)
        """
        # Resize ให้ตรงกับตอนเทรน (256x256)
        img = cv2.resize(img_array, (256, 256))
        
        # Normalize (0-1)
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # เพิ่มมิติ (Batch Size, Channels, H, W) -> (1, 1, 256, 256)
        return img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def calculate_loss(self, base64_string):
        """
        ฟังก์ชันหลักที่ API เรียกใช้
        Input: Base64 String
        Output: Loss (float)
        """
        try:
            # 1. จัดการ Base64 Header (ถ้ามี)
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            # 2. Decode Base64 -> Bytes -> Numpy
            img_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # อ่านเป็น Grayscale
            
            if img is None:
                raise ValueError("Could not decode image")

            # 3. Preprocess & Inference
            input_tensor = self.preprocess(img)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                # คำนวณ MSE Loss
                loss = torch.mean((input_tensor - output) ** 2).item()
            
            return loss

        except Exception as e:
            print(f"Error in calculate_loss: {e}")
            # ส่งค่า Loss สูงๆ หรือ -1 เพื่อบอกว่า Error (แล้วแต่ Logic ปลายทาง)
            return -1.0

    def check_is_lung(self, img_bytes):
        """
        (เผื่อไว้ใช้) ตรวจสอบว่าเป็นภาพปอดหรือไม่ตาม Threshold
        """
        # แปลง bytes เป็น numpy ก่อนเรียก preprocess
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        input_tensor = self.preprocess(img)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            loss = torch.mean((input_tensor - output) ** 2).item()
        
        is_valid = loss <= self.threshold
        return is_valid, loss