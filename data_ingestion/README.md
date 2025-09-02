# Data ingestion for cis warehouse

## Overview
This project feeds data from various data sources into cis_war db for future processing  

---

## Features
- **face_features**: ingest data created by face_features (donor photo clasification) project. 
- **czech_cities**: ingest data created by cities locations, Postal Codes and GPS locations.
- **zts_list**: ingest czech list of plasmapheresis concurents, maintained by regulators. The code compares online zts list with the one saved in s3, if it fails it throws error, in this case file in s3 must be updated, if is the change not big edit the zts_list.json in s3 manually, check lib/zts.py for details
---

## Requirements
- Docker & Docker Compose
- MinIO instance (can run via Docker)
- CIS db replica:

---

## Environment Variables
Create a `.env.sample` file in the project root:

## Running
from analytics root directory, run
```bash
python -m data_ingestion.main
```

## Dev
C:/Users/lukas/AppData/Local/Programs/Python/Python311/Scripts/uvicorn.exe app.main:app --host 0.0.0.0 --port 8000

## To add new feature/ingestion
1) Place the source to stable feedable location (minio, samba), add env.S3_FILE... for maintainability. Direct internet links are not recommended/unstable
2) Add folder to data_sources, add schema.py here
3) Add file to loaders folder, place whole processing there. Use other files here as started. Keep convention run to run the ingestion
4) Add this file to main.py

## Running the Tests

Run all tests:
cd data_ingestion
python -m pytest

Run specific test file:
python -m pytest tests/test_population_density_loader.py

Run with coverage:
python -m pytest --cov=data_ingestion

Verbose output:
python -m pytest -v