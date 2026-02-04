import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from typing import List
from app.services.ai_engine import process_medical_images

app = FastAPI(
    title="Medical AI Image Service",
    description="Service สำหรับวิเคราะห์ Change Map จากภาพ CXR 2 ระยะเวลา",
    version="1.1.0"
)

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ready", "service": "ai-engine"}

@app.post("/api/v1/analyze", tags=["AI Pipeline"])
async def analyze_cxr(
    image1: UploadFile = File(..., description="ภาพ CXR ใบที่ 1 (อดีต)"),
    image2: UploadFile = File(..., description="ภาพ CXR ใบที่ 2 (ปัจจุบัน)"),
    roi_data: str = Form(..., description="JSON string ของ ROI เช่น {'roi1': [x,y,w,h], 'roi2': [x,y,w,h]}")
):
    """
    Pipeline: 
    1. Validation 
    2. One-class Check 
    3. Normalization 
    4. Segmentation 
    5. Registration 
    6. Change Map Generation
    """
    
    # 1. Validation: ตรวจสอบประเภทไฟล์
    allowed_types = ["image/jpeg", "image/png", "image/dicom"] # เพิ่ม DICOM ถ้าจำเป็น
    if image1.content_type not in allowed_types or image2.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Unsupported file type. Please upload JPG or PNG."
        )

    # 2. Parse ROI JSON
    try:
        parsed_roi = json.loads(roi_data)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid ROI JSON format."
        )

    try:
        # 3. Read Files
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        # 4. Execute Pipeline (ส่งงานต่อไปยัง Service Layer)
        # เราจะไม่เขียน Logic การประมวลผลที่นี่ เพื่อให้โค้ดทดสอบง่าย
        result = process_medical_images(img1_bytes, img2_bytes, parsed_roi)

        # 5. Handle Business Logic Errors
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=result["error"]
            )

        return result

    except Exception as e:
        # Log error ที่นี่ (เช่นใช้ Loguru หรือ Logging)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal Server Error: {str(e)}"
        )
    
    finally:
        # คืนทรัพยากร Memory
        await image1.close()
        await image2.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)