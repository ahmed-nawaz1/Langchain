from pymongo import MongoClient

from dotenv import load_dotenv
import os 

load_dotenv()

mongodbUrl = os.getenv("MONGO_DB")