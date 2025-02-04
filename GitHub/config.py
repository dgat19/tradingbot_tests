import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FINANCIAL_PREP_API_KEY = os.getenv("FINANCIAL_PREP_API_KEY")
MONGO_DB_USER = os.getenv("MONGO_DB_USER")
MONGO_DB_PASS = os.getenv("MONGO_DB_PASS")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = os.getenv("BASE_URL")
