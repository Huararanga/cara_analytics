import pandas as pd

from data_ingestion.data_sources.zts_list.schema import ZTSList
from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.utils.file_loader import read_from_minio
from data_ingestion.db.postgres import get_engine

def run(S3_BUCKET, S3_FILE):
    engine = get_engine()
    df = pd.read_json(read_from_minio(S3_BUCKET, S3_FILE));

    loader = BaseLoader(ZTSList, engine)
    loader.load(df)
