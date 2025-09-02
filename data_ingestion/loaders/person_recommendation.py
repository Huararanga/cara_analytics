import pandas as pd

from data_ingestion.data_sources.person_recommendation.schema import PersonRecommendation
from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.db.postgres import get_engine
from data_ingestion.db.central import get_central_engine

def run():
    engine = get_engine();
    central_engine = get_central_engine();

    query = """
        SELECT 
            cr.id_recommendation,
            cl.hemo_id AS id_person,
            clr.hemo_id AS id_person_recommendation,
            cl.id_client AS id_client,
            clr.id_client AS id_client_recommendation,
            cr.referral AS referral,
            cr.success_closed AS success_closed,
            cr.closed_date
        FROM CP.client_recommendation cr
        JOIN CP.clients cl ON cr.client = cl.id_client
        JOIN CP.clients clr ON cr.recommendation_client = clr.id_client
        # WHERE clr.hemo_id IS NOT NULL AND cl.hemo_id IS NOT NULL;
    """
    df = pd.read_sql(query, con=central_engine)

    df["referral"] = df["referral"].astype(bool)
    df["success_closed"] = df["success_closed"].astype(bool)
    df["closed_date"] = df["closed_date"].replace("0000-00-00 00:00:00", pd.NA)
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

    loader = BaseLoader(PersonRecommendation, engine)
    loader.load(df)
