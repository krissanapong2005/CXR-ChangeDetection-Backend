import mysql.connector
from mysql.connector import Error
from app.config import settings 

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Database Connection Error: {e}")
        return None

def test_connection():
    conn = get_db_connection()
    if conn:
        print(f"MySQL ({settings.DB_HOST}) Connected Successfully!")
        conn.close()
    else:
        print(f"MySQL ({settings.DB_HOST}) Connection Failed!")