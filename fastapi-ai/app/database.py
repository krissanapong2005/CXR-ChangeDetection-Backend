import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """เชื่อมต่อกับ MySQL Server ของ Software Engineer Project"""
    try:
        connection = mysql.connector.connect(
            host='...',    # IP จากข้อมูลโปรเจกต์
            port=3306,             # Port มาตรฐาน
            user='...',            # Username
            password='...', # Password
            database='...'      # ชื่อ DB ที่เราจะใช้งาน (ตรวจสอบกับทีมอีกครั้ง)
        )
        return connection
    except Error as e:
        print(f"Connection Failed to 10.224.8.12: {e}")
        return None