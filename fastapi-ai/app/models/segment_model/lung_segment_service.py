import torch
import torchvision
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os
from modelStructure import PretrainedUNet
import base64
class LungSegmentationService:
    def __init__(self, model_path, device=None):
        # ตั้งค่า Device อัตโนมัติถ้าไม่ได้ระบุ
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. นิยามโครงสร้าง Model (ต้องมั่นใจว่า Class PretrainedUNet ถูก Import มาแล้ว)
        self.model = PretrainedUNet(
            in_channels=1,
            out_channels=3,
            batch_norm=True
        )
        
        self.model_path = model_path
        self._load_weights() # โหลด Weight ทันทีเมื่อสร้าง Instance

    def _load_weights(self):
        """โหลด Weight ของโมเดลเข้าสู่ Memory"""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, image_path, size=(256, 256)):
        """
        ทำนายผลจากไฟล์รูปภาพ
        Returns:
            pred_overlay: รูปภาพที่ทำ Overlay แล้ว (Numpy Array)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"ไม่เจอไฟล์ที่ {image_path}")

        # 1. Pre-processing
        img = Image.open(image_path).convert("L")
        img_resized = TF.resize(img, size)
        input_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 3. Post-processing & Visualization
        image_np = np.array(img_resized)
        image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        # สร้าง Overlay (ทำเป็น Vectorized operation เพื่อความเร็ว)
        mask_layer = np.zeros_like(image_color)
        mask_layer[pred_mask == 1] = [255, 0, 0] # ซ้าย: แดง
        mask_layer[pred_mask == 2] = [0, 255, 0] # ขวา: เขียว
        
        pred_overlay = cv2.addWeighted(image_color, 0.6, mask_layer, 0.4, 0)
        cropped_result = self.get_cropped_image(image_color, pred_mask)

        return pred_overlay, cropped_result
    
    def get_cropped_image(self, original_image, pred_mask, padding=15):
            binary_mask = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
            coords = cv2.findNonZero(binary_mask)
            
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                
                # --- แก้ไขตรงนี้ ---
                # ถ้าส่ง image_color เข้ามา (NumPy Array)
                # shape จะได้เป็น (height, width, channels)
                orig_h, orig_w = original_image.shape[:2] 
                
                mask_h, mask_w = pred_mask.shape
                
                scale_x = orig_w / mask_w
                scale_y = orig_h / mask_h
                
                xmin = max(0, int((x - padding) * scale_x))
                ymin = max(0, int((y - padding) * scale_y))
                xmax = min(orig_w, int((x + w + padding) * scale_x))
                ymax = min(orig_h, int((y + h + padding) * scale_y))

                # การ Crop ใน NumPy ต้องใช้ [y1:y2, x1:x2]
                cropped_img = original_image[ymin:ymax, xmin:xmax]
                
                return cropped_img
            else:
                return None

    def prepare_base64_for_predict(b64_string, target_size=(256, 256)):
        """
        แปลง Base64 เป็น Tensor ที่พร้อมส่งเข้า model.predict()
        """
        # 1. ตัด header ถ้ามี (เช่น data:image/png;base64,...)
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        # 2. Decode และแปลงเป็น Numpy Array
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # อ่านเป็น Grayscale
        
        if image is None:
            return None, None

        # 3. Pre-processing (ทำเหมือนตอนเทรน)
        # เก็บภาพต้นฉบับไว้ทำ visualization หรือผลลัพธ์
        original_size = image.shape[:2] 
        image_resized = cv2.resize(image, target_size)
        
        # แปลงเป็น Tensor และ Normalize [0, 1]
        input_tensor = TF.to_tensor(image_resized) # กลายเป็น (1, H, W)
        input_tensor = input_tensor.unsqueeze(0)   # เพิ่มมิติ Batch -> (1, 1, H, W)
        
        return input_tensor, original_size

    def image_to_base64(image_np):
        """
        แปลงรูปภาพ (Numpy Array) เป็น Base64 String
        image_np: สามารถเป็นได้ทั้งภาพ Grayscale, BGR (ภาพสี) หรือภาพที่ Overlay แล้ว
        """
        # 1. เลือกนามสกุลไฟล์ที่ต้องการ (แนะนำ .jpg สำหรับภาพถ่าย หรือ .png สำหรับภาพที่ต้องการความคมชัด)
        # ในกรณี Overlay หรือ Cropped แนะนำ .jpg เพื่อให้ขนาดไฟล์ไม่ใหญ่เกินไป
        success, buffer = cv2.imencode('.jpg', image_np)
        
        if not success:
            return None
        
        # 2. แปลงเป็น Base64
        b64_string = base64.b64encode(buffer).decode('utf-8')
        
        # 3. (Optional) เติม Prefix เพื่อให้ Frontend นำไปแสดงผลได้ทันที
        return b64_string