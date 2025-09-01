import pandas as pd

from data_ingestion.data_sources.face_features.schema import PersonPhotoFaceFeatures
from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.utils.file_loader import read_from_minio
from data_ingestion.db.postgres import get_engine

def run(S3_BUCKET, S3_FILE):
    engine = get_engine()
    df = pd.read_parquet(read_from_minio(S3_BUCKET, S3_FILE));

    loader = BaseLoader(PersonPhotoFaceFeatures, engine)
    # skip embedding until we have use for it
    df = df.drop(columns=["embedding"])
    loader.load(df)
