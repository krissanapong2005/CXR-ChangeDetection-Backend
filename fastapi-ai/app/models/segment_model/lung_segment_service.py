# lung_segment_service.py
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
import os
import base64
from .modelStructure import PretrainedUNet  # ตรวจสอบว่าไฟล์นี้มีอยู่จริง

class LungSegmentationService:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. นิยามโครงสร้าง Model
        self.model = PretrainedUNet(
            in_channels=1,
            out_channels=3,
            batch_norm=True
        )
        
        self.model_path = model_path
        self._load_weights()

    def _load_weights(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_mask(self, base64_string):
        """
        ฟังก์ชันหลักสำหรับ API: รับ Base64 -> ทำนาย -> ตัดภาพ -> คืนค่า Base64
        """
        # 1. แปลง Base64 เป็น Tensor และได้ภาพ Numpy เดิมกลับมาด้วย (เพื่อเอาไว้ Crop)
        input_tensor, original_image_np = self.prepare_base64_for_predict(base64_string)
        
        if input_tensor is None:
            raise ValueError("Invalid Base64 Image String")

        # 2. Inference (ทำนายผล)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            # แปลงผลลัพธ์เป็น Mask (0, 1, 2)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 3. Crop ภาพปอดตาม Mask ที่ได้
        cropped_img_np = self.get_cropped_image(original_image_np, pred_mask)

        if cropped_img_np is None:
            # กรณีหาปอดไม่เจอ ให้คืนรูปเดิมกลับไป หรือจะ return None ก็ได้
            print("Lung not found, returning original image.")
            return self.image_to_base64(original_image_np)

        # 4. แปลงภาพที่ Crop แล้วกลับเป็น Base64 เพื่อส่งต่อให้ส่วนอื่น
        return self.image_to_base64(cropped_img_np)

    def get_cropped_image(self, original_image, pred_mask, padding=15):
        binary_mask = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
        coords = cv2.findNonZero(binary_mask)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            orig_h, orig_w = original_image.shape[:2] 
            mask_h, mask_w = pred_mask.shape
            
            scale_x = orig_w / mask_w
            scale_y = orig_h / mask_h
            
            xmin = max(0, int((x - padding) * scale_x))
            ymin = max(0, int((y - padding) * scale_y))
            xmax = min(orig_w, int((x + w + padding) * scale_x))
            ymax = min(orig_h, int((y + h + padding) * scale_y))

            cropped_img = original_image[ymin:ymax, xmin:xmax]
            return cropped_img
        else:
            return None

    def prepare_base64_for_predict(self, b64_string, target_size=(256, 256)):
        """
        แปลง Base64 เป็น Tensor และคืนค่า Numpy Array ของภาพเดิม
        """
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        try:
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            # อ่านเป็น Grayscale เพื่อเข้า Model แต่ตอน Crop อาจจะอยากได้ภาพสีหรือไม่ก็ได้
            # ในที่นี้ขออ่านเป็น Grayscale ตาม Model UNet (in_channels=1)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                return None, None

            # เก็บภาพต้นฉบับไว้ (Numpy)
            original_image_np = image.copy()

            # Resize เพื่อเข้า Model
            image_resized = cv2.resize(image, target_size)
            
            # Normalize & ToTensor
            input_tensor = TF.to_tensor(image_resized) 
            input_tensor = input_tensor.unsqueeze(0) 
            
            return input_tensor, original_image_np
        except Exception as e:
            print(f"Base64 Error: {e}")
            return None, None

    def image_to_base64(self, image_np):
        """
        แปลงรูปภาพ Numpy กลับเป็น Base64
        """
        try:
            success, buffer = cv2.imencode('.jpg', image_np)
            if not success:
                return None
            b64_string = base64.b64encode(buffer).decode('utf-8')
            return b64_string
        except Exception as e:
            print(f"Encoding Error: {e}")
            return None