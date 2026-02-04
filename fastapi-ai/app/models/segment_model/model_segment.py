import torch
import torchvision
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os

class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out

class UNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)

    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)

    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.enc1 = Block(in_channels, 64, 64, batch_norm)
        self.enc2 = Block(64, 128, 128, batch_norm)
        self.enc3 = Block(128, 256, 256, batch_norm)
        self.enc4 = Block(256, 512, 512, batch_norm)

        self.center = Block(512, 1024, 512, batch_norm)

        self.dec4 = Block(1024, 512, 256, batch_norm)
        self.dec3 = Block(512, 256, 128, batch_norm)
        self.dec2 = Block(256, 128, 64, batch_norm)
        self.dec1 = Block(128, 64, 64, batch_norm)

        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))
        enc4 = self.enc4(self.down(enc3))

        center = self.center(self.down(enc4))

        dec4 = self.dec4(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))

        out = self.out(dec1)

        return out

class PretrainedUNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)

    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)

    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.init_conv = torch.nn.Conv2d(in_channels, 3, 1)

        endcoder = torchvision.models.vgg11(pretrained=True).features
        self.conv1 = endcoder[0]   # 64
        self.conv2 = endcoder[3]   # 128
        self.conv3 = endcoder[6]   # 256
        self.conv3s = endcoder[8]  # 256
        self.conv4 = endcoder[11]   # 512
        self.conv4s = endcoder[13]  # 512
        self.conv5 = endcoder[16]  # 512
        self.conv5s = endcoder[18] # 512

        self.center = Block(512, 512, 256, batch_norm)

        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)

        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        enc1 = torch.nn.functional.relu(self.conv1(init_conv), inplace=True)
        enc2 = torch.nn.functional.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3s(enc3), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4s(enc4), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5s(enc5), inplace=True)

        center = self.center(self.down(enc5))

        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))

        out = self.out(dec1)

        return out
    
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