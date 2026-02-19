from app.utils import decode_base64_to_bytes, numpy_to_base64
from app.models.one_class_model import AnomalyDetector
from app.models.segment_model import lung_segment_service, normalize_contrast_service
from app.services.registration import Img_registration
from app.services.change_detection import block_based_pca_change_detection

model_path = r''
# โหลด Detector เตรียมไว้ (Singleton Pattern)
detector = AnomalyDetector(threshold=0.001949) # ใช้ค่าที่คุณคำนวณจาก Colab
segment_service = lung_segment_service(model_path)
registration = Img_registration()
norm_service = normalize_contrast_service()

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

    # --- ขั้นตอนที่ทำ: Normalize ---
    norm_img = norm_service.process_and_normalize(img1_bytes, img2_bytes, roi_data["target"], roi_data["source"])

    # --- ขั้นตอนที่ทำ: Segment & Crop ---
    #convert base64 to np
    img1, _ = segment_service.prepare_byte_for_predict(img1_bytes)
    img2, _ = segment_service.prepare_byte_for_predict(img2_bytes)
    #convert np to base64
    pred_overlay1, crop_img1 = segment_service.predict(img1)
    pred_overlay2, crop_img2 = segment_service.predict(img2)

    # --- ขั้นตอนที่ทำ: Image Registration ---
    img_aligned = registration.align_images_ecc_with_resize(crop_img1, crop_img2)
    if img_aligned is None:
        return {"error": "ECC ล้มเหลว (ภาพอาจแตกต่างกันเกินไป)"}

    # --- ขั้นตอนที่ทำ: Change detection ---
    change_map = block_based_pca_change_detection(crop_img1, img_aligned)
    change_map_base64 = numpy_to_base64(change_map)

    # 3. ส่งผลลัพธ์กลับเป็น Base64 เพื่อให้ Laravel/Frontend แสดงผล <--- พาร์ทนี้ยังไม่นิ่ง
    return {
        "status": "success",
        "result_image": change_map_base64, # ใช้ numpy_to_base64(change_map_np)
    }