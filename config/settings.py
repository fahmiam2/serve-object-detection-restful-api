from dotenv import load_dotenv
import os

load_dotenv()

GCS_KEY_FILE = os.getenv("GCS_KEY_FILE")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")