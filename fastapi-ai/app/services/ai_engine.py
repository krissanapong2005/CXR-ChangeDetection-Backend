import os
import torch
from pathlib import Path
from app.database import get_db_connection
from app.models.segment_model.lung_segment_service import LungSegmentationService
from app.models.one_class_model.detector import AnomalyDetector

# --- Robust Path Configuration ---
# 1. Get the directory where THIS file (ai_engine.py) is located
BASE_DIR = Path(__file__).resolve().parent.parent # Goes up to 'app/' folder

# 2. Define absolute paths to your models
# Adjust these paths to match your actual folder structure
SEGMENT_MODEL_PATH = BASE_DIR / "models" / "segment_model" / "lung_segmentation_model50epoch.pth"
ONE_CLASS_MODEL_PATH = BASE_DIR / "models" / "one_class_model" / "weights_v1.pth"

# Convert to strings for the loaders
SEGMENT_MODEL_PATH = str(SEGMENT_MODEL_PATH)
ONE_CLASS_MODEL_PATH = str(ONE_CLASS_MODEL_PATH)

# --- Singleton Model Loading ---
try:
    segmenter = LungSegmentationService(SEGMENT_MODEL_PATH)
    detector = AnomalyDetector(ONE_CLASS_MODEL_PATH)
    print(f"Successfully loaded models from: \n1. {SEGMENT_MODEL_PATH}\n2. {ONE_CLASS_MODEL_PATH}")
except Exception as e:
    print(f"Critical Error loading models: {e}")
    # Consider raising an error here so the server doesn't start with broken models
    segmenter = None
    detector = None

def process_medical_images(img1_base64: str, img2_base64: str, roi_data: dict):
    if segmenter is None or detector is None:
        return {"error": "AI Models not initialized. Check server logs for path errors."}

    try:
        # 1. Run Lung Segmentation
        mask1 = segmenter.get_mask(img1_base64)
        mask2 = segmenter.get_mask(img2_base64)

        # 2. Run Anomaly Detection (Calculate Loss)
        loss1 = detector.calculate_loss(img1_base64)
        loss2 = detector.calculate_loss(img2_base64)

        # 3. Save to MySQL
        is_saved = save_analysis_to_db(loss1, loss2)

        return {
            "status": "success",
            "db_status": "saved" if is_saved else "failed",
            "analysis": {
                "is_lung": True,
                "img1_loss": float(loss1),
                "img2_loss": float(loss2)
            },
            "output_images": {
                "image1_cropped": img1_base64, 
                "image2_cropped": img2_base64
            },
            "roi_info": roi_data
        }
    except Exception as e:
        return {"error": f"AI Runtime Error: {str(e)}"}

def save_analysis_to_db(l1, l2):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO history (result_loss_1, result_loss_2, created_at) VALUES (%s, %s, NOW())"
            cursor.execute(query, (l1, l2))
            conn.commit()
            return True
        except Exception as e:
            print(f"SQL Error: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    return False