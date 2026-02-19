import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d

def block_based_pca_change_detection(img_target, img_aligned, block_size=3):
    """
    ตรวจจับการเปลี่ยนแปลงโดยใช้ Block-based PCA เพื่อดูบริบทพิกเซลข้างเคียง
    """
    # 1. โหลดภาพและ Pre-process
    I1 = img_target
    I2 = img_aligned
    h, w = I1.shape
    I2 = cv.resize(I2, (w, h))

    # 2. สกัด Patch (บล็อกย่อย) จากทั้งสองภาพ
    # patch_size=(3,3) จะทำให้ 1 พิกเซลมีข้อมูลรอบตัว 9 จุด
    patches1 = extract_patches_2d(I1, (block_size, block_size))
    patches2 = extract_patches_2d(I2, (block_size, block_size))

    # 3. สร้าง Data Matrix D
    # คลี่แต่ละ Patch เป็นเวกเตอร์แถว (Flat) แล้วเอามาต่อกัน
    # D จะมีขนาด (จำนวนพิกเซล, block_size * block_size * 2)
    p1_flat = patches1.reshape(patches1.shape[0], -1)
    p2_flat = patches2.reshape(patches2.shape[0], -1)
    D = np.hstack((p1_flat, p2_flat))

    # 4. ทำ PCA
    # เราสนใจส่วนประกอบหลักที่ 1 (โครงสร้างร่วม) และที่เหลือคือการเปลี่ยนแปลง
    pca = PCA(n_components=1)
    D_transformed = pca.fit_transform(D)

    # คำนวณ Error (Residual) ระหว่างข้อมูลจริงกับข้อมูลที่ถูกสร้างใหม่จาก PC1
    # ส่วนที่ PCA อธิบายไม่ได้ (Reconstruction Error) คือ "การเปลี่ยนแปลง"
    D_reconstructed = pca.inverse_transform(D_transformed)
    # diff = np.linalg.norm(D - D_reconstructed, axis=1)
    diff_vector = D[:, block_size**2:] - D_reconstructed[:, block_size**2:]
    change_direction = np.mean(diff_vector, axis=1) # หาค่าเฉลี่ยในบล็อกนั้น

    # 5. แปลงผลลัพธ์กลับเป็นภาพ
    # เนื่องจาก extract_patches_2d จะทำให้ขอบภาพหายไปเล็กน้อย (ตามขนาด block)
    # เราจึงต้องคำนวณขนาดภาพใหม่
    new_h = h - block_size + 1
    new_w = w - block_size + 1
    # change_map = diff.reshape(new_h, new_w)
    change_map = change_direction.reshape(new_h, new_w)
    v_max = np.max(np.abs(change_map))
    norm_map = ((change_map + v_max) / (2 * v_max) * 255).astype(np.uint8)
    color_mapped = cv.applyColorMap(norm_map, cv.COLORMAP_JET)

    return color_mapped