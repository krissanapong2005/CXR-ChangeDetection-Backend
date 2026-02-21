import torch
import cv2
import numpy as np
import os
from .architecture import Autoencoder
from app.utils import decode_base64_to_bytes

class AnomalyDetector:
    def __init__(self, model_filename="weights_v1.pth", threshold=None):
        # 1. ตั้งค่าพื้นฐาน
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold if threshold else 0.001949
        
        # 2. โหลดโมเดล
        self.model = Autoencoder().to(self.device)
        
        # ค้นหา Path ของไฟล์ weights แบบ Dynamic
        base_path = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_path, model_filename)
        
        # โหลดน้ำหนักลงในโมเดล
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, img):
        # แปลง bytes เป็น numpy array และ resize ให้ตรงกับตอนเทรน (256x256)
        # nparr = np.frombuffer(img_bytes, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        
        # Normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def check_is_lung(self, img_bytes):
        """รับ Base64 แล้วตรวจสอบว่าเป็นภาพปอดไหม"""       
        # ทำ Preprocess (Resize เป็น 256x256 และ Normalize 0-1)
        input_tensor = self.preprocess(img_bytes)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            loss = torch.mean((input_tensor - output) ** 2).item()
        
        is_valid = loss <= self.threshold
        return is_valid, loss