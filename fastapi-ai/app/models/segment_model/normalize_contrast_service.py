import numpy as np
import cv2
import base64

class normalizeService():
    def __init__(self):
        pass

    def _base64_to_cv2(self, b64_string):
        """แปลง Base64 string ให้เป็น OpenCV image (numpy array)"""
        try:
            # ตัด header data:image/png;base64,... ออกถ้ามี
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # อ่านเป็น Grayscale ตามโค้ดเดิมของคุณ
            return img
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None

    def _cv2_to_base64(self, image):
        """แปลง OpenCV image ให้เป็น Base64 string"""
        _, buffer = cv2.imencode('.png', image)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return b64_string

    def process_and_normalize(self, target_b64, source_b64, roi_target, roi_source):
        """ฟังก์ชันหลักสำหรับรับ-ส่ง Base64"""
        
        # 1. แปลง Input Base64 -> Numpy Array
        # image_target = self._base64_to_cv2(target_b64)
        # image_source = self._base64_to_cv2(source_b64)
        image_target = target_b64
        image_source = source_b64

        if image_target is None or image_source is None:
            return None

        # 2. คำนวณ Normalization (Logic เดิมของคุณ)
        y_start_t, y_end_t, x_start_t, x_end_t = roi_target
        roi_t_data = image_target[y_start_t:y_end_t, x_start_t:x_end_t]
        mu_target, sigma_target = np.mean(roi_t_data), np.std(roi_t_data)

        y_start_s, y_end_s, x_start_s, x_end_s = roi_source
        roi_s_data = image_source[y_start_s:y_end_s, x_start_s:x_end_s]
        mu_source, sigma_source = np.mean(roi_s_data), np.std(roi_s_data)

        if sigma_source == 0:
            return source_b64 # คืนค่าเดิมถ้าปรับไม่ได้

        normalized = image_source.astype(float)
        standardized = (normalized - mu_source) / sigma_source
        final_image = mu_target + sigma_target * standardized
        
        result_img = np.clip(final_image, 0, 255).astype(np.uint8)

        # 3. แปลงผลลัพธ์กลับเป็น Base64
        # return self._cv2_to_base64(result_img)
        return result_img