from sqlalchemy import Column, Integer, Float

from data_ingestion.data_sources.config import Base

class PopulationDensity(Base):
    __tablename__ = 'population_density'
    id = Column(Integer, primary_key=True)
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    density = Column(Float, nullable=False)