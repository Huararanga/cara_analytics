# main.py
import os
from dotenv import load_dotenv

from data_ingestion.data_sources.config import Base
from data_ingestion.utils.file_loader import initS3
from data_ingestion.db.postgres import initPostgres
from data_ingestion.db.central import initCentral
from data_ingestion.loaders import face_features, czech_cities, zts_list, person_recommendation

load_dotenv()

DB_URL = os.getenv("DB_URL", "REQUIRED")
CENTRAL_DB_URL = os.getenv("CENTRAL_DB_URL", "REQUIRED")

S3_BUCKET = os.getenv("S3_BUCKET", "data_ingestion")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:19090")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_FILE_FACE_FEATURES = os.getenv("S3_FILE_FACE_FEATURES", "face_features.parquet")
S3_FILE_CZECH_CITIES = os.getenv("S3_FILE_CZECH_CITIES", "face_features.parquet")
S3_FILE_ZTS_LIST = os.getenv("S3_FILE_ZTS_LIST", "zts_list.json")

def main():
    initS3(S3_ENDPOINT,S3_ACCESS_KEY, S3_SECRET_KEY);
    initPostgres(DB_URL);
    initCentral(CENTRAL_DB_URL);

    face_features.run(S3_BUCKET, S3_FILE_FACE_FEATURES);
    czech_cities.run(S3_BUCKET, S3_FILE_CZECH_CITIES);
    
    # TODO: call to refresh ztslist
    # ztsList = zts.load_zts_list_from_db(engine)
    # # and log changes
    zts_list.run(S3_BUCKET, S3_FILE_ZTS_LIST);

    person_recommendation.run()

if __name__ == "__main__":
    main()
