#utils.py
import base64
import numpy as np
import cv2
import io
from PIL import Image

def decode_base64_to_bytes(base64_string: str) -> bytes:
    """
    ถอดรหัส Base64 string เป็น bytes โดยรองรับทั้งแบบมีและไม่มี Data Header
    """
    try:
        if "," in base64_string:
            # ตัดส่วน 'data:image/png;base64,' ออกถ้ามี
            base64_string = base64_string.split(",")[1]
        
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"ไม่สามารถถอดรหัส Base64 ได้: {str(e)}")

def encode_bytes_to_base64(img_bytes: bytes, format: str = "png") -> str:
    """
    แปลง bytes กลับเป็น Base64 string เพื่อส่งกลับไปแสดงผลบนหน้าเว็บ
    """
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format};base64,{encoded}"

def numpy_to_base64(img_np: np.ndarray) -> str:
    """
    แปลงภาพจาก OpenCV (Numpy) เป็น Base64 โดยตรง
    """
    _, buffer = cv2.imencode('.png', img_np)
    return encode_bytes_to_base64(buffer.tobytes())