# main.py
from dotenv import load_dotenv
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Tuple
from app.database.connect_database import test_connection
from app.services.ai_engine import process_medical_images
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import pyotp


app = FastAPI(
    title="Medical AI Image Service",
    description="Service สำหรับวิเคราะห์ Change Map จากภาพ CXR ผ่าน Base64",
    version="1.2.0"
)


load_dotenv()
SHARED_SECRET = os.getenv("SHARED_SECRET")

# print(f"[*] Checking .env at: {ENV_PATH}")
# if os.getenv("SHARED_SECRET"):
#     print("✅ SUCCESS: API_SHARED_SECRET is loaded!")
# else:
#     print("❌ STILL MISSING: API_SHARED_SECRET is not found.")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = exc.errors()
    missing_fields = [err["loc"][-1] for err in error_details if err["type"] == "missing"]
    
    if missing_fields:
        error_msg = f"ข้อมูลที่ส่งมาไม่ครบถ้วน{', '.join(missing_fields)}"
    else:
        error_msg = "รูปแบบข้อมูลที่ส่งมาไม่ถูกต้อง (Validation Error)"

    return JSONResponse(
        status_code=422,
        content={
            "status": "fail",
            "message": error_msg
        }
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

def verify_rolling_key(x_api_key: str = Header(...)):
    # ถ้า SHARED_SECRET เป็น None ให้โยน Error 500 แจ้ง Admin ทันที
    if not SHARED_SECRET:
        print("Error: API_SHARED_SECRET is not set in environment variables.")
        raise HTTPException(status_code=500, detail="Server Auth Configuration Error")
    
    try:
        totp = pyotp.TOTP(SHARED_SECRET)
        if not totp.verify(x_api_key):
            raise HTTPException(status_code=401, detail="Key expired or invalid")
    except Exception as e:
        print(f"pyotp error: {e}")
        raise HTTPException(status_code=401, detail="Invalid Key Format")
    return True

@app.on_event("startup")
async def startup_event():
    print("Starting CXR Analysis System...")
    test_connection()

@app.get("/")
def read_root():
    return {"message": "CXR Backend is running!"}

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ready", "service": "ai-engine"}

@app.post("/api/v1/analyze", tags=["AI Pipeline"], dependencies=[Depends(verify_rolling_key)])
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