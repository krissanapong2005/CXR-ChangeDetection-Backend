from app.utils import decode_base64_to_bytes, numpy_to_base64
from app.models.one_class_model import AnomalyDetector

# โหลด Detector เตรียมไว้ (Singleton Pattern)
detector = AnomalyDetector(threshold=0.001949) # ใช้ค่าที่คุณคำนวณจาก Colab

def process_medical_images(img1_base64: str, img2_base64: str, roi_data: dict):
    # 1. แปลง Base64 เป็น Bytes
    try:
        img1_bytes = decode_base64_to_bytes(img1_base64)
        img2_bytes = decode_base64_to_bytes(img2_base64)
    except ValueError as e:
        return {"error": str(e)}

    # 2. ส่งเข้า One-class Model ด่านแรก
    is_valid1, error1 = detector.check_is_lung(img1_bytes)
    is_valid2, error2 = detector.check_is_lung(img2_bytes)
    
    if not is_valid1 or not is_valid2:
        return {"error": "คุณภาพของภาพไม่ผ่านการตรวจสอบ (Anomaly Detected)"}

    # --- ขั้นตอนที่ทำ: Segment & Crop ---
    # สมมติว่าได้ผลลัพธ์เป็นภาพ Numpy Array จากขั้นตอนถัดๆ ไป
    # change_map_np = ...

    # 3. ส่งผลลัพธ์กลับเป็น Base64 เพื่อให้ Laravel/Frontend แสดงผล <--- พาร์ทนี้ยังไม่นิ่ง
    # return {
    #     "status": "success",
    #     "result_image": "base64_string_here", # ใช้ numpy_to_base64(change_map_np)
    #     "metrics": {"error1": error1, "error2": error2}
    # }