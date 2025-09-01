1. Develop nb vscode
2. Run it "todo how to from console"
3. export by "jupyter-nbconvert ZubovaKrizovka.ipynb --to html"

## Data ingestion
This project feeds data from various data sources into cis_war db for future processing

To run:
```bash
python -m data_ingestion.main
```

## Experiments

Experiments are executed using marimo
```bash
marimo exploration.py
```

## Face features app

Extracts face features from images from cis person.photos and store them into parquet

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```