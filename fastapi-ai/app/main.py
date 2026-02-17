import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Router ที่เราสร้างไว้ในไฟล์ endpoints.py
from app.api.endpoints import router as ai_router

# 1. สร้าง Instance ของ App
app = FastAPI(
    title="Medical AI Image Service",
    description="Service สำหรับวิเคราะห์ Change Map จากภาพ CXR ผ่าน Base64",
    version="1.2.0"
)

# 2. ตั้งค่า CORS (เพื่อให้ Frontend/Laravel เรียกใช้งานได้)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ใน Production ควรระบุ Domain ของ Laravel เช่น ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"], # อนุญาตทุก Method (GET, POST, etc.)
    allow_headers=["*"], # อนุญาตทุก Header
)

# 3. ลงทะเบียน Router (Include Router)
# กำหนด prefix เป็น /api/v1 ดังนั้น Endpoint จริงจะเป็น: POST /api/v1/analyze
app.include_router(ai_router, prefix="/api/v1", tags=["AI Pipeline"])

# 4. System Health Check
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ready", "service": "ai-engine"}

@app.get("/")
async def root():
    return {"message": "Medical AI Service is running."}

# 5. Config สำหรับการรัน (Entry Point)
if __name__ == "__main__":
    # app.main:app หมายถึง ไฟล์ app/main.py และตัวแปรชื่อ app
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)