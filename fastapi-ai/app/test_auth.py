import requests
import pyotp
import time

# 1. ตั้งค่า Secret ให้ตรงกับใน .env ของ Backend
SHARED_SECRET = "JBSWY3DPEHPK3PXP" # หรือใช้ค่าที่คุณเจนไว้
BASE_URL = "http://127.0.0.1:8000"

def test_analyze_api():
    # 2. สร้าง Rolling Key (TOTP) ณ เวลาปัจจุบัน
    totp = pyotp.TOTP(SHARED_SECRET)
    current_otp = totp.now()
    print(f"[*] Generated OTP: {current_otp}")

    # 3. เตรียมข้อมูล Request Payload (ตาม AnalyzeRequest schema)
    payload = {
        "image1_base64": "test_image_1_data",
        "image2_base64": "test_image_2_data",
        "roi": {
            "target": {"xy_start": [10.0, 10.0], "xy_end": [100.0, 100.0]},
            "source": {"xy_start": [20.0, 20.0], "xy_end": [120.0, 120.0]}
        }
    }

    # 4. ยิง Request โดยแนบ x-api-key ใน Header
    headers = {
        "x-api-key": current_otp,
        "Content-Type": "application/json"
    }

    print("[*] Sending request to /api/v1/analyze...")
    response = requests.post(f"{BASE_URL}/api/v1/analyze", json=payload, headers=headers)

    # 5. ตรวจสอบผลลัพธ์
    if response.status_code == 200:
        print("✅ Success: Access Granted!")
        print(f"Response: {response.json()}")
    elif response.status_code == 401:
        print("❌ Failed: Unauthorized (Key expired or invalid)")
    else:
        print(f"⚠️ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_analyze_api()