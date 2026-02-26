from app.utils import decode_base64_to_bytes, numpy_to_base64, byte_to_np_grayscale
from app.models.one_class_model import AnomalyDetector
from app.models.segment_model import lung_segment_service
from app.services.registration import Img_registration
from app.services.change_detection import block_based_pca_change_detection
from app.models.segment_model.normalize_contrast_service import normalizeService
import os

model_path = os.path.join('app', 'models', 'segment_model', 'lung_segmentation_model50epoch.pth')

# โหลด Detector เตรียมไว้ (Singleton Pattern)
try:
    detector = AnomalyDetector(threshold=0.001949) # ใช้ค่าที่คุณคำนวณจาก Colab
    segment_service = lung_segment_service.LungSegmentationService(model_path)
    registration = Img_registration()
    norm_service = normalizeService()
except Exception as e:
    print(f"Model Loading Error: {e}")

def process_medical_images(img1_base64: str, img2_base64: str, roi_data: dict):
    try:
        # 1. แปลง Base64 เป็น Bytes
        try:
            img1_bytes = decode_base64_to_bytes(img1_base64)
            img2_bytes = decode_base64_to_bytes(img2_base64)
            image1 = byte_to_np_grayscale(img1_bytes)
            image2 = byte_to_np_grayscale(img2_bytes)
        except ValueError as e:
            return {"status": "fail", "message": str(e)}

        # 2. ตรวจสอบคุณภาพภาพ
        is_valid1, error1 = detector.check_is_lung(image1)
        is_valid2, error2 = detector.check_is_lung(image2)
        
        if not is_valid1 or not is_valid2:
            return {"status": "fail", "message": "คุณภาพของภาพไม่ผ่านการตรวจสอบ (Anomaly Detected)"}

        # 3. Normalize
        try:
            norm_img = norm_service.process_and_normalize(image1, image2, roi_data["target"], roi_data["source"])
        except Exception as e:
            return {"status": "fail", "message": str(e)}

        # 4. Segment & Crop
        try:
            img1_tensor = segment_service.prepare_for_predict(image1)
            img2_tensor = segment_service.prepare_for_predict(norm_img)
            
            crop_img1 = segment_service.predict(img1_tensor, image1)
            crop_img2 = segment_service.predict(img2_tensor, norm_img)
        except Exception as e:
            return {"status": "fail", "message": str(e)}

        if crop_img1 is None or crop_img2 is None:
            return {"status": "fail", "message": "ไม่สามารถตรวจจับขอบเขตปอดเพื่อตัดภาพได้"}

        # 5. Image Registration
        try:
            img_aligned = Img_registration.align_images_ecc_with_resize(crop_img1, crop_img2)
        except Exception as e:
            return {"status": "fail", "message": str(e)}
            
        if img_aligned is None:
            return {"status": "fail", "message": "กระบวนการจัดวางภาพ (Registration) ล้มเหลว"}

        # 6. Change detection
        try:
            change_map = block_based_pca_change_detection(crop_img1, img_aligned)
            
            # แปลงกลับเป็น Base64 เพื่อส่งออก
            change_map_base64 = numpy_to_base64(change_map)
            crop_img1_base64 = numpy_to_base64(crop_img1)
            crop_img2_base64 = numpy_to_base64(crop_img2)
        except Exception as e:
            return {"status": "fail", "message": str(e)}

        return {
            "status": "success",
            "message": "ok",
            "image_source": crop_img1_base64,
            "image_target": crop_img2_base64,
            "image_changeMap": change_map_base64
        }

    except Exception as e:
        return {"status": "fail", "message": str(e)}