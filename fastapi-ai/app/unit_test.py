import pytest
import pyotp
import os
from fastapi.testclient import TestClient
from main import app # สมมติว่าไฟล์หลักชื่อ main.py

# จำลองการตั้งค่า Environment Variable สำหรับ Test
os.environ["API_SHARED_SECRET"] = "JBSWY3DPEHPK3PXP"
client = TestClient(app)

def test_auth_success():
    """ทดสอบกรณีส่ง Key ถูกต้อง"""
    totp = pyotp.TOTP(os.getenv("API_SHARED_SECRET"))
    valid_key = totp.now()
    
    response = client.post(
        "/api/v1/analyze",
        json={
            "image1_base64": "data", "image2_base64": "data",
            "roi": {
                "target": {"xy_start": [0,0], "xy_end": [1,1]},
                "source": {"xy_start": [0,0], "xy_end": [1,1]}
            }
        },
        headers={"x-api-key": valid_key}
    )
    # หมายเหตุ: จะได้ 200 หรือ 422 ขึ้นอยู่กับ ai_engine ข้างใน แต่ต้องไม่ใช่ 401
    assert response.status_code != 401

def test_auth_failed_invalid_key():
    """ทดสอบกรณีส่ง Key ผิด"""
    response = client.post(
        "/api/v1/analyze",
        json={},
        headers={"x-api-key": "000000"} # Key มั่ว
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Key expired or invalid"

def test_auth_missing_header():
    """ทดสอบกรณีไม่ส่ง Header มาเลย"""
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422 # FastAPI จะตีกลับเพราะขาด Required Header