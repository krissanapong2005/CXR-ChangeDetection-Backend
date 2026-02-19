import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Img_registration:
    def __init__(self, img1=None, img2=None):
        self.image_target = img1
        self.image_moving = img2

    def resize_keep_aspect_ratio(image, target_width=None):
        """ย่อภาพโดยรักษาอัตราส่วน (Aspect Ratio) ตามความกว้างที่กำหนด"""
        if target_width is None:
            return image
        
        (h, w) = image.shape[:2]
        r = target_width / float(w)
        dim = (target_width, int(h * r))
        
        # ใช้ INTER_AREA ซึ่งเหมาะสำหรับการย่อภาพ (Downsampling)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    def align_images_ecc_with_resize(image_target, image_moving, processing_width=800, number_of_iterations=100):
        img_fixed_bgr = image_target
        img_moving_bgr = image_moving

        # --- 3. [ขั้นตอนใหม่] ปรับขนาดภาพ (Resizing) ---
        # 3.1 ย่อภาพ Fixed ให้มีความกว้างตามที่กำหนด (เพื่อความเร็วในการคำนวณ)
        # img_fixed_resized = resize_keep_aspect_ratio(img_fixed_bgr, target_width=processing_width)
        # img_fixed_resized = img_fixed_bgr
        
        # 3.2 ปรับภาพ Moving ให้มีขนาด (w, h) เท่ากับภาพ Fixed ที่ย่อแล้วเป๊ะๆ
        # ข้อควรระวัง: findTransformECC ต้องการให้ภาพทั้งสองมีขนาดเท่ากัน (Dimension Matching)
        target_h, target_w = img_fixed_bgr.shape[:2]
        img_moving_resized = cv2.resize(img_moving_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # print(f"ปรับขนาดภาพเพื่อประมวลผลเป็น: {target_w}x{target_h}")

        # --- 4. เตรียมภาพสำหรับ ECC (Grayscale) ---
        img_fixed_gray = cv2.cvtColor(img_fixed_bgr, cv2.COLOR_BGR2GRAY)
        img_moving_gray = cv2.cvtColor(img_moving_resized, cv2.COLOR_BGR2GRAY)

        # --- 5. ตั้งค่า ECC ---
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        print("กำลังคำนวณ ECC... (Speed up by resizing)")
        
        try:
            # --- 6. คำนวณหา Transformation Matrix ---
            (cc, warp_matrix) = cv2.findTransformECC(img_fixed_gray, img_moving_gray, warp_matrix, warp_mode, criteria)
            print(f"ECC สำเร็จ (Correlation: {cc:.4f})")
        except cv2.error as e:
            print("Error: ECC ล้มเหลว (ภาพอาจแตกต่างกันเกินไป)")
            return

        # --- 7. ใช้ Warp Affine ---
        sz = img_fixed_gray.shape
        # ใช้ WARP_INVERSE_MAP เพราะ ECC map Fixed -> Moving
        img_aligned = cv2.warpAffine(img_moving_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return img_aligned