import pandas as pd

from data_ingestion.data_sources.population_density.schema import PopulationDensity
from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.utils.file_loader import read_from_minio
from data_ingestion.db.postgres import get_engine

def run(S3_BUCKET, S3_FILE):
    engine = get_engine()
    df = pd.read_csv(read_from_minio(S3_BUCKET, S3_FILE))
    
    # Rename columns to match schema
    df = df.rename(columns={"X": "longitude", "Y": "latitude", "Z": "density"})
    
    loader = BaseLoader(PopulationDensity, engine)
    loader.load(df)