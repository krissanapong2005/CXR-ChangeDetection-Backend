from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any

# Import ฟังก์ชันประมวลผลหลักจาก Service
# (ต้องมั่นใจว่ามีไฟล์ app/services/ai_engine.py อยู่จริง)
from app.services.ai_engine import process_medical_images

# สร้าง Router
router = APIRouter()

class AnalyzeRequest(BaseModel):
    image1_base64: str  # ภาพอดีต
    image2_base64: str  # ภาพปัจจุบัน
    roi_data: Dict[str, Any] # ข้อมูล ROI เช่น {"roi1": [x,y,w,h], ...}

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_cxr(request: AnalyzeRequest):
    """
    API Endpoint สำหรับวิเคราะห์ภาพถ่ายรังสีทรวงอก (CXR)
    - รับ Base64 ของภาพ 2 ภาพ และข้อมูล ROI
    - ส่งกลับผลลัพธ์การวิเคราะห์หรือ Error Message
    """
    
    # 1. Validation: ตรวจสอบว่ามีการส่งข้อมูลภาพมาจริงหรือไม่
    if not request.image1_base64 or not request.image2_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="กรุณาส่งข้อมูลภาพในรูปแบบ Base64 ให้ครบทั้ง 2 ภาพ"
        )

    try:
        # 2. Call Service: เรียกใช้ Logic การประมวลผล AI
        result = process_medical_images(
            request.image1_base64, 
            request.image2_base64, 
            request.roi_data
        )

        # 3. Handle Logical Errors: ตรวจสอบ Error จาก AI Engine (เช่น ไม่ใช่ภาพปอด)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=result["error"]
            )

        # 4. Success: ส่งผลลัพธ์กลับ
        return result

    except HTTPException as he:
        # ส่งต่อ HTTP Exception ที่เราสร้างไว้เอง
        raise he
    except Exception as e:
        print(f"Internal Server Error: {e}") # ควรเปลี่ยนเป็น logging ใน production
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"เกิดข้อผิดพลาดภายในระบบ: {str(e)}"
        )