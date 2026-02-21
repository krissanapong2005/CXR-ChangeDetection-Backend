from app.utils import decode_base64_to_bytes, numpy_to_base64, byte_to_np_grayscale
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
    """
    image1 คือ image_target, image2 คือ image_sorce
    image_target จะถูก crop อย่างเดียว
    image_sorce จะถูก normให้ตรงกับ target, crop และ ทำ registration
    """
    # 1. แปลง Base64 เป็น Bytes
    try:
        img1_bytes = decode_base64_to_bytes(img1_base64)
        img2_bytes = decode_base64_to_bytes(img2_base64)
        image1 = byte_to_np_grayscale(img1_bytes)
        image2 = byte_to_np_grayscale(img2_bytes)
    except ValueError as e:
        return {"error": str(e)}

    # 2. ส่งเข้า One-class Model ด่านแรก
    is_valid1, error1 = detector.check_is_lung(image1)
    is_valid2, error2 = detector.check_is_lung(image2)
    
    if not is_valid1 or not is_valid2:
        return {"error": "คุณภาพของภาพไม่ผ่านการตรวจสอบ (Anomaly Detected)"}

    # --- ขั้นตอนที่ทำ: Normalize ---
    norm_img = norm_service.process_and_normalize(image1, image2, roi_data["target"], roi_data["source"])

    # --- ขั้นตอนที่ทำ: Segment & Crop ---
    #convert base64 to np
    img1_tensor = segment_service.prepare_for_predict(image1)
    img2_tensor = segment_service.prepare_for_predict(norm_img)
    #convert np to base64
    crop_img1 = segment_service.predict(img1_tensor, image1)
    crop_img2 = segment_service.predict(img2_tensor, norm_img)

    # --- ขั้นตอนที่ทำ: Image Registration ---
    img_aligned = registration.align_images_ecc_with_resize(crop_img1, crop_img2)
    if img_aligned is None:
        return {"error": "ECC ล้มเหลว (ภาพอาจแตกต่างกันเกินไป)"}

    # --- ขั้นตอนที่ทำ: Change detection ---
    change_map = block_based_pca_change_detection(crop_img1, img_aligned)
    change_map_base64 = numpy_to_base64(change_map)
    crop_img1_base64 = numpy_to_base64(crop_img1)
    crop_img2_base64 = numpy_to_base64(crop_img2)

    # 3. ส่งผลลัพธ์กลับเป็น Base64 เพื่อให้ Laravel/Frontend แสดงผล <--- พาร์ทนี้ยังไม่นิ่ง
    return {
        "status": "success",
        "result_image": change_map_base64,
        "crop_img1": crop_img1_base64,
        "crop_img2": crop_img2_base64
    }