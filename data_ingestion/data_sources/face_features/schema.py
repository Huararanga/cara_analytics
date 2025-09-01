from sqlalchemy import (
    Column, Integer, Float, String, Boolean
)
from data_ingestion.data_sources.config import Base

class PersonPhotoFaceFeatures(Base):
    __tablename__ = 'person_photos_face_features'

    id_photo = Column(Integer, primary_key=True)
    id_person = Column(Float, nullable=True)  # Nullable due to possible NaNs
    firstname = Column(String, nullable=True)
    surname = Column(String, nullable=True)
    face_detected = Column(Boolean, nullable=True)
    age = Column(Float, nullable=True)
    gender = Column(String, nullable=True)
    dominant_emotion = Column(String, nullable=True)
    pose_yaw = Column(Float, nullable=True)
    pose_pitch = Column(Float, nullable=True)
    pose_roll = Column(Float, nullable=True)

    # Embedding skip this for now
    # embedding = Column(Text, nullable=True)

    emotion_angry = Column(Float, nullable=True)
    emotion_disgust = Column(Float, nullable=True)
    emotion_fear = Column(Float, nullable=True)
    emotion_happy = Column(Float, nullable=True)
    emotion_sad = Column(Float, nullable=True)
    emotion_surprise = Column(Float, nullable=True)
    emotion_neutral = Column(Float, nullable=True)
