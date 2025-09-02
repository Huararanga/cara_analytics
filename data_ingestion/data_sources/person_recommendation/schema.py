from sqlalchemy import Column, Integer, Boolean, DateTime

from data_ingestion.data_sources.config import Base

class PersonRecommendation(Base):
    __tablename__ = 'person_recommendation'  #comes from central.client_recommendation
    id_recommendation = Column(Integer, primary_key=True)
    id_person = Column(Integer, nullable=True)
    id_person_recommendation = Column(Integer, nullable=True)
    id_client = Column(Integer, nullable=False)
    id_client_recommendation = Column(Integer, nullable=False)
    referral = Column(Boolean, nullable=False)
    success_closed = Column(Boolean, nullable=False)
    closed_date = Column(DateTime, nullable=True)
