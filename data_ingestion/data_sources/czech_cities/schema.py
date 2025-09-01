from sqlalchemy import Column, Integer, Float, String

from data_ingestion.data_sources.config import Base

class CzechCities(Base):
    __tablename__ = 'czech_cities'
    id = Column(Integer, primary_key=True)
    country = Column(String, nullable=True)
    postal_code = Column(String, nullable=True)
    place_name = Column(String, nullable=True)
    admin_name1 = Column(String, nullable=True)
    admin_code1 = Column(Integer, nullable=True)
    admin_name2 = Column(String, nullable=True)
    admin_code2 = Column(Integer, nullable=True)
    admin_name3 = Column(Float, nullable=True)   # originally object, but only NaN seen; kept as Float
    admin_code3 = Column(Float, nullable=True)   # float64 due to NaN
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    accuracy = Column(Integer, nullable=True)
