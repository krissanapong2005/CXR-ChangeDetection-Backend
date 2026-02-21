import numpy as np
import cv2
import base64

class normalizeService():
    def __init__(self):
        pass

    def process_and_normalize(self, image_target, image_source, roi_target, roi_source):
        """
        roi format --> roi:{target:{xy_start:(float,float),xy_end:(float,float)},
                            source:{xy_start:(float,float),xy_end:(float,float)}}
        x, y are in percentage 0-1
        """
        # 1. แปลง Input Base64 -> Numpy Array
        target_h, target_w = image_target.shape[:2] 
        source_h, source_w = image_source.shape[:2] 

        if image_target is None or image_source is None:
            return None

        # 2. คำนวณ Normalization (Logic เดิมของคุณ)
        # ดึงพิกัด Normalized (0-1) ออกมา
        norm_x_start_t, norm_y_start_t = roi_target['xy_start']
        norm_x_end_t, norm_y_end_t = roi_target['xy_end']

        # คำนวณเป็นพิกเซล (Integer) โดยจับคู่ x-Width และ y-Height ให้ถูกต้อง
        x_start_t = int(norm_x_start_t * target_w)
        y_start_t = int(norm_y_start_t * target_h)
        x_end_t = int(norm_x_end_t * target_w)
        y_end_t = int(norm_y_end_t * target_h)
        roi_t_data = image_target[y_start_t:y_end_t, x_start_t:x_end_t]
        mu_target, sigma_target = np.mean(roi_t_data), np.std(roi_t_data)

        # ดึงพิกัด Normalized (0-1) ออกมา
        norm_x_start_s, norm_y_start_s = roi_source['xy_start']
        norm_x_end_s, norm_y_end_s = roi_source['xy_end']

        # คำนวณเป็นพิกเซล (Integer) โดยจับคู่ x-Width และ y-Height ให้ถูกต้อง
        x_start_s = int(norm_x_start_s * source_w)
        y_start_s = int(norm_y_start_s * source_h)
        x_end_s = int(norm_x_end_s * source_w)
        y_end_s = int(norm_y_end_s * source_h)
        roi_s_data = image_source[y_start_s:y_end_s, x_start_s:x_end_s]
        mu_source, sigma_source = np.mean(roi_s_data), np.std(roi_s_data)

        if sigma_source == 0:
            return image_source # คืนค่าเดิมถ้าปรับไม่ได้

        normalized = image_source.astype(float)
        standardized = (normalized - mu_source) / sigma_source
        final_image = mu_target + sigma_target * standardized
        
        result_img = np.clip(final_image, 0, 255).astype(np.uint8)

        return result_img