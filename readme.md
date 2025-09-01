# Setup

## Prerequisites
- Python 3.12+ installed
- Install python3-venv package if not available:
  ```bash
  sudo apt install python3.12-venv  # Ubuntu/Debian
  ```

## Virtual Environment Setup
1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies (if requirements.txt exists):
   ```bash
   pip install -r requirements.txt
   ```

## Jupyter Notebook Development
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