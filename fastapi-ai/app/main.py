from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Tuple

# นำเข้าฟังก์ชันจากโครงสร้างโฟลเดอร์ที่คุณวางไว้
from app.services.ai_engine import process_medical_images

app = FastAPI(
    title="Medical AI Image Service",
    description="Service สำหรับวิเคราะห์ Change Map จากภาพ CXR ผ่าน Base64",
    version="1.2.0"
)

# 1. กำหนดโครงสร้างพิกัด (float, float)
class ROIEntry(BaseModel):
    xy_start: Tuple[float, float]
    xy_end: Tuple[float, float]

# 2. กำหนดโครงสร้าง ROI ที่มี target และ source
class ROIData(BaseModel):
    target: ROIEntry
    source: ROIEntry

# 3. ปรับ AnalyzeRequest ให้ใช้ ROIData
class AnalyzeRequest(BaseModel):
    image1_base64: str
    image2_base64: str 
    roi: ROIData

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ready", "service": "ai-engine"}

@app.post("/api/v1/analyze", tags=["AI Pipeline"])
async def analyze_cxr(request: AnalyzeRequest):
    """
    Pipeline สำหรับ Base64:
    1. รับ JSON Payload (Base64 + ROI)
    2. ส่งต่อไปยัง ai_engine เพื่อ Decode และประมวลผล
    3. รับผลลัพธ์กลับมาเป็น JSON (มีภาพผลลัพธ์เป็น Base64)
    """
    try:
        # 2. ตรวจสอบเบื้องต้นว่ามีการส่ง String มาจริงหรือไม่
        if not request.image1_base64 or not request.image2_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="กรุณาส่งข้อมูลภาพในรูปแบบ Base64 ทั้ง 2 ภาพ"
            )

        # 3. ส่งข้อมูลไปยัง Service Layer (ai_engine.py)
        # เนื่องจากเป็น Base64 string อยู่แล้ว จึงไม่ต้องใช้ 'await .read()' เหมือนไฟล์
        result = process_medical_images(
            request.image1_base64, 
            request.image2_base64, 
            request.roi.model_dump()
        )

        # 4. ตรวจสอบ Business Logic Error (เช่น One-class ไม่ผ่าน)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=result["error"]
            )

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        # ในระดับ Production ควรใช้ Logging เก็บรายละเอียด Error จริง
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"เกิดข้อผิดพลาดภายในระบบ: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)