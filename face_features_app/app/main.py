# app/main.py
import os
import io
import json
import base64
import boto3
import pandas as pd
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine
from pathlib import Path
from insightface.app import FaceAnalysis
from deepface import DeepFace
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# -------------------
# ENV CONFIG
# -------------------

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "face-features")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:19090")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_FILE = os.getenv("S3_FILE", "face_features.parquet")

# -------------------
# AWS S3 Client (MinIO)
# -------------------
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# -------------------
# DB Engine
# -------------------
engine = create_engine(os.getenv("DB_URL"))

# -------------------
# FaceAnalysis Singleton
# -------------------
_face_app = None
def get_face_app():
    global _face_app
    if _face_app is None:
        providers = ort.get_available_providers()
        device = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
        _face_app = FaceAnalysis(name="buffalo_l", providers=device)
        _face_app.prepare(ctx_id=0)
    return _face_app

# -------------------
# Helpers
# -------------------
def base64_to_image(b64):
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        b64 = b64.replace('\n', '').replace('\r', '').strip()
        missing_padding = len(b64) % 4
        if missing_padding:
            b64 += '=' * (4 - missing_padding)
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def load_existing_from_s3():
    """Load existing parquet file from S3 (MinIO) if available."""
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_FILE)
        print(f"✅ Loaded existing {S3_FILE} from S3 ({S3_BUCKET})")
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            print(f"⚠ File {S3_FILE} not found in S3 bucket {S3_BUCKET}. Returning empty DataFrame.")
            return pd.DataFrame()
        elif code == "NoSuchBucket":
            print(f"⚠ Bucket {S3_BUCKET} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=S3_BUCKET)
            return pd.DataFrame()
        else:
            raise


def save_to_s3(df: pd.DataFrame):
    """Save DataFrame to S3 in Parquet format."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=S3_FILE, Body=buf.getvalue())
    print(f"💾 Saved {len(df)} records to {S3_FILE} in S3 bucket {S3_BUCKET}")


def s3_file_exists():
    """Check if file exists in S3."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=S3_FILE)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        else:
            raise

def fetch_photos_from_db(processed_ids=None):
    if processed_ids:
        ids_str = ",".join(map(str, processed_ids))
        query = f"""
            SELECT
                pho.id as id_photo,
                per.id as id_person,
                per.firstname,
                per.surname,
                pho.content
            FROM donor.person as per
            LEFT JOIN donor.photo as pho ON per.id = pho.id_donor
            WHERE pho.id NOT IN ({ids_str});
        """
    else:
        query = """
            SELECT
                pho.id as id_photo,
                per.id as id_person,
                per.firstname,
                per.surname,
                pho.content
            FROM donor.person as per
            LEFT JOIN donor.photo as pho ON per.id = pho.id_donor;
        """
    return pd.read_sql(query, engine)

def extract_features(df_photos):
    face_app = get_face_app()
    results = []
    for _, row in df_photos.iterrows():
        img = base64_to_image(row["content"])
        if img is None:
            results.append({"id_photo": int(row["id_photo"]), "face_detected": False})
            continue
        faces = face_app.get(img)
        if not faces:
            results.append({"id_photo": int(row["id_photo"]), "face_detected": False})
            continue
        face = faces[0]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cropped_face = img[y1:y2, x1:x2]
        try:
            emo_result = DeepFace.analyze(
                cropped_face,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )
            dominant_emotion = emo_result[0]["dominant_emotion"]
            emotion_probs = emo_result[0]["emotion"]
        except Exception:
            dominant_emotion = None
            emotion_probs = {}
        record = {
            "id_photo": int(row["id_photo"]),
            "id_person": int(row["id_person"]),
            "firstname": row["firstname"],
            "surname": row["surname"],
            "face_detected": True,
            "age": face.age,
            "gender": "female" if face.sex == 0 else "male",
            "dominant_emotion": dominant_emotion,
            "pose_yaw": face.pose[0],
            "pose_pitch": face.pose[1],
            "pose_roll": face.pose[2],
            "embedding": face.embedding.tolist()
        }
        for emo_label, emo_score in emotion_probs.items():
            record[f"emotion_{emo_label}"] = float(emo_score)
        results.append(record)
    return pd.DataFrame(results)

# -------------------
# FastAPI App
# -------------------
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/update")
def update_photos():
    existing_df = load_existing_from_s3()
    processed_ids = set(existing_df["id_photo"].tolist()) if not existing_df.empty else None
    df_photos = fetch_photos_from_db(processed_ids)
    if df_photos.empty:
        return {"message": "No new photos to process."}
    new_features = extract_features(df_photos)
    combined_df = pd.concat([existing_df, new_features], ignore_index=True)
    save_to_s3(combined_df)
    return {"message": f"Processed {len(new_features)} new photos."}

@app.post("/reset")
def reset_photos():
    df_photos = fetch_photos_from_db()
    features_df = extract_features(df_photos)
    save_to_s3(features_df)
    return {"message": f"Processed all {len(features_df)} photos."}

@app.get("/get-file")
def get_file():
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_FILE)
        return obj["Body"].read()
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="File not found.")

@app.post("/get-fields")
def get_fields(fields: list[str] = Query(...)):
    df = load_existing_from_s3()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available.")
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")
    df = df[fields]
    # Lower numeric precision
    for col in df.select_dtypes(include=[np.float64, np.float32]).columns:
        df[col] = df[col].round(4)
    return json.loads(df.to_json(orient="records"))
