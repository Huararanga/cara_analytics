import pandas as pd

from data_ingestion.data_sources.czech_cities.schema import CzechCities
from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.utils.file_loader import read_from_minio
from data_ingestion.db.postgres import get_engine

def run(S3_BUCKET, S3_FILE):
    engine = get_engine()
    df = pd.read_csv(read_from_minio(S3_BUCKET, S3_FILE), sep="\t", header=None);
    df.columns = [
        "country", "postal_code", "place_name", "admin_name1", "admin_code1",
        "admin_name2", "admin_code2", "admin_name3", "admin_code3",
        "latitude", "longitude", "accuracy"
    ]
    df["postal_code"] = df["postal_code"].str.replace(" ", "", regex=False)


    loader = BaseLoader(CzechCities, engine)
    loader.load(df)
