from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints import router
from app.database import get_db_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # สิ่งที่ทำตอนเปิด Server
    print("Starting CXR Analysis System...")
    conn = get_db_connection()
    if conn:
        print("MySQL (10.224.8.12) Connected")
        conn.close()
    yield
    # สิ่งที่ทำตอนปิด Server
    print("Shutting down...")

app = FastAPI(title="CXR Change Detection API", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")