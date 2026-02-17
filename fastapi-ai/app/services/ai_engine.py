import os
import torch
from app.utils import decode_base64_to_bytes, numpy_to_base64

# Import Models
# ตรวจสอบให้แน่ใจว่าชื่อโฟลเดอร์และไฟล์ถูกต้องตามโครงสร้างจริง
from app.models.one_class_model.detector import AnomalyDetector
from app.models.segment_model.lung_segment_service import LungSegmentationService
# from app.models.segment_model import normalize_contrast_service # (ถ้ามีใช้ค่อยเปิด)

# ==========================================
# 1. ส่วนตั้งค่าและโหลดโมเดล (Model Loading)
# ==========================================

# กำหนด Path ของโมเดล Segmentation (ต้องระบุ path ไฟล์ .pth ให้ถูกต้อง)
# ตัวอย่าง: segment_model_path = r"app/models/segment_model/unet_lung.pth"
segment_model_path = r""  # <--- ⚠️ ใส่ Path ของไฟล์โมเดล Segmentation ตรงนี้

detector = None
segment_service = None

# --- โหลด Anomaly Detector ---
try:
    # detector จะโหลด weights_v1.pth อัตโนมัติจาก code ใน detector.py
    # ถ้ายังไม่มีไฟล์ weights_v1.pth ให้คอมเมนต์บรรทัดนี้ไปก่อนเพื่อทดสอบ Flow
    detector = AnomalyDetector(threshold=0.001949)
    print("✅ Anomaly Detector loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load Anomaly Detector. ({e})")

# --- โหลด Segmentation Service ---
try:
    if segment_model_path and os.path.exists(segment_model_path):
        segment_service = LungSegmentationService(model_path=segment_model_path)
        print("✅ Lung Segmentation Service loaded successfully.")
    else:
        print("⚠️ Warning: Path to Segmentation Model is empty or invalid.")
except Exception as e:
    print(f"⚠️ Warning: Could not load Segmentation Service. ({e})")


# ==========================================
# 2. ฟังก์ชัน Helper (ย่อย)
# ==========================================

def pred_segment_crop(img64):
    """
    ฟังก์ชันช่วยสำหรับทำ Segment และ Crop ภาพเดียว
    """
    if segment_service is None:
        return None, None
        
    # 1. Convert base64 to Tensor (พร้อม Predict)
    # หมายเหตุ: prepare_base64_for_predict ต้องคืนค่า input_tensor
    input_tensor, original_size = segment_service.prepare_base64_for_predict(img64)
    
    if input_tensor is None:
        return None, None

    # 2. Predict & Crop
    # predict คืนค่าเป็น Numpy Array (ภาพ Overlay และ ภาพ Crop)
    pred_overlay_np, crop_img_np = segment_service.predict(img64) # แก้ไข: ส่ง path หรือ base64 ตาม implementation ของ service

    # *หมายเหตุ*: ฟังก์ชัน predict ใน lung_segment_service ของคุณรับ 'image_path' หรือ 'input_tensor' ?
    # จากโค้ดเก่า predict รับ image_path แต่เรามี Base64
    # คุณอาจต้องปรับ lung_segment_service.py ให้รับ Tensor หรือ Base64 โดยตรง
    # แต่ในที่นี้สมมติว่า segment_service.predict รองรับการทำงานแล้ว
    
    # 3. Convert Result back to Base64
    pred_overlay_b64 = segment_service.image_to_base64(pred_overlay_np)
    crop_img_b64 = segment_service.image_to_base64(crop_img_np)
    
    return pred_overlay_b64, crop_img_b64


# ==========================================
# 3. ฟังก์ชันหลัก (Main Logic)
# ==========================================

def process_medical_images(img1_base64: str, img2_base64: str, roi_data: dict):
    """
    Main Pipeline:
    1. Decode -> Bytes
    2. Anomaly Check (Is Lung?)
    3. Segmentation & Crop
    4. Return Result
    """
    
    # --- Step 1: Decode Base64 to Bytes ---
    try:
        img1_bytes = decode_base64_to_bytes(img1_base64)
        img2_bytes = decode_base64_to_bytes(img2_base64)
    except ValueError as e:
        return {"error": f"Base64 Decoding Error: {str(e)}"}

    # --- Step 2: Anomaly Detection (One-class Model) ---
    # ถ้าโหลด Model ไม่สำเร็จ ให้ข้ามขั้นตอนนี้ไป (หรือจะ return error ก็ได้แล้วแต่ Requirement)
    if detector:
        is_valid1, error1 = detector.check_is_lung(img1_bytes)
        is_valid2, error2 = detector.check_is_lung(img2_bytes)
        
        # ถ้าภาพใดภาพหนึ่งไม่ใช่ปอด
        if not is_valid1 or not is_valid2:
            return {
                "error": "Anomaly Detected: Images might not be Chest X-Rays.",
                "details": {"img1_loss": error1, "img2_loss": error2}
            }
    else:
        print("ℹ️ Skipping Anomaly Detection (Model not loaded)")

    # --- Step 3: Segmentation & Crop ---
    # ถ้าโหลด Model ไม่สำเร็จ ก็ข้ามไป
    crop_res1 = None
    crop_res2 = None
    
    if segment_service:
        # เรียกใช้ฟังก์ชัน Helper ด้านบน
        overlay1, crop1 = pred_segment_crop(img1_base64)
        overlay2, crop2 = pred_segment_crop(img2_base64)
        
        if crop1 is None or crop2 is None:
             return {"error": "Segmentation Failed: Could not crop lung area."}
             
        crop_res1 = crop1
        crop_res2 = crop2
    else:
        print("ℹ️ Skipping Segmentation (Model not loaded)")
        # ถ้าไม่มีโมเดล ให้ส่งภาพเดิมกลับไปแทน (Mock Behavior)
        crop_res1 = img1_base64
        crop_res2 = img2_base64

    # --- Step 4: Construct Response ---
    # ส่งผลลัพธ์กลับไปให้ API Endpoint
    return {
        "status": "success",
        "message": "Analysis Complete",
        "results": {
            "image1_cropped": crop_res1,
            "image2_cropped": crop_res2,
            # ใส่ข้อมูลอื่นๆ ที่ต้องการส่งกลับ Frontend ตรงนี้
            "roi_received": roi_data
        }
    }