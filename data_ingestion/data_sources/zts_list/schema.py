from sqlalchemy import Column, Integer, Float, String

from data_ingestion.data_sources.config import Base

class ZTSList(Base):
    __tablename__ = 'zts_list'

    name = Column(String, nullable=True)
    ICO = Column(Integer, nullable=True)
    id_SUKL = Column(Integer, nullable=True)
    clinic_code = Column(String, primary_key=True)
    city = Column(String, nullable=True)
    company = Column(String, nullable=True)
    city_postal_code = Column(String, nullable=True)
    city_latitude = Column(Float, nullable=True)
    city_longitude = Column(Float, nullable=True)
