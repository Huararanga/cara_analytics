# Face Features Extraction API

## Overview
This project extracts facial features (embedding, age, gender, emotion probabilities, pose) from images stored in a database, stores results in **Parquet format** on **MinIO (S3)**, and serves them via a **FastAPI** service.  

The system:
- Uses **InsightFace** for embeddings, age, gender, and pose.
- Uses **DeepFace** for emotion analysis.
- Reads base64-encoded images from a PostgreSQL database.
- Saves results as a **Parquet file** to a locally hosted **MinIO** bucket.
- Avoids reprocessing already-processed images.
- Provides an API to update, reset, download, and query processed features.

---

## Features
- ** /update Incremental Updates**: Processes only new images when using `/update`.
- ** /reset Full Reset**: Reprocesses all images from scratch using `/reset`.
- **S3 Storage**: Stores the `face_features.parquet` file in MinIO.
- **Data Query**: Returns JSON output with selected fields and reduced numeric precision for lightweight usage.
- `/get-file`: Download Parquet file from S3.
- `/get-fields`: Get selected fields with reduced numeric precision.
- **Dockerized**: Fully containerized for easy deployment.
- **`/health`**: check the server.
---

## Requirements
- Docker & Docker Compose
- Local MinIO instance (can run via Docker)
- PostgreSQL database with:
  - `donor.person` table (`id`, `firstname`, `surname`)
  - `donor.photo` table (`id_donor`, `content` - base64 image)

---

## Environment Variables
Create a `.env` file in the project root:

```env
# Database
DB_URL=postgresql+psycopg2://user:password@db_host:5432/database_name

# MinIO
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=face-features
S3_FILE=face_features.parquet
```

## Running with Docker
not tested
```bash
docker build -t face-features-service .
docker run --rm -p 8000:8000 --env-file=.env face-features-service
```

## Dev
works
C:/Users/lukas/AppData/Local/Programs/Python/Python311/Scripts/uvicorn.exe app.main:app --host 0.0.0.0 --port 8000