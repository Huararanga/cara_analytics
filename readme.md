# Cara Analytics

Plasma donation data analysis and machine learning platform for donor behavior prediction and clinic optimization.

## Project Structure

```
cara_analytics/
├── lib/                    # Core analysis modules (installed as package)
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── geo.py             # Geographic analysis tools
│   ├── hmm.py             # Hidden Markov Models for donor behavior
│   ├── lost_donors.py     # Lost donor prediction models
│   └── zts.py             # Competitor analysis tools
├── locations/             # Location and population data modules
│   └── population_density/ # Population mapping tools
├── experiment/            # Marimo notebooks for analysis
│   └── draw.py           # Main analysis notebook
├── data_ingestion/       # Data pipeline from various sources
├── face_features_app/    # Face recognition feature extraction
├── pyproject.toml        # Modern Python package configuration
└── .venv/               # Virtual environment
```

## Setup

### Prerequisites
- Python 3.12+ installed
- Install python3-venv package if not available:
  ```bash
  sudo apt install python3.12-venv  # Ubuntu/Debian
  ```

### Installation
1. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate on Windows
   ```

2. Install project in editable mode (installs all dependencies):
   ```bash
   pip install -e .
   ```

   This installs the `cara-analytics` package and all its dependencies from `pyproject.toml`.

## Dependency Management

### Adding New Dependencies
1. Add dependencies directly to `pyproject.toml`:
   ```toml
   dependencies = [
       "new-package>=1.0.0",
       # ... other dependencies
   ]
   ```

2. Reinstall in editable mode to pick up changes:
   ```bash
   pip install -e .
   ```

### Development Dependencies
For testing, linting, and development tools:
```bash
pip install -e ".[dev]"  # Installs dev dependencies
```

### Package Structure Benefits
- **Clean imports**: `from lib import preprocessing as prep`
- **No sys.path hacks**: Proper Python package structure
- **Editable development**: Changes reflect immediately
- **Dependency tracking**: All dependencies in one place

## Development Workflow

### Interactive Analysis with Marimo
Run the main analysis notebook:
```bash
source .venv/bin/activate
marimo edit experiment/draw.py
```

### Development Best Practices
1. **Environment**: Always activate virtual environment before working
2. **Dependencies**: Add new packages to `pyproject.toml`, not ad-hoc installs
3. **Imports**: Use clean package imports: `from lib import module_name`
4. **Testing**: Run tests before committing changes: `pytest`

### Code Organization
- **lib/**: Reusable analysis functions and classes
- **experiment/**: Interactive notebooks for exploration and visualization
- **locations/**: Geographic data processing modules
- **data_ingestion/**: ETL pipelines for data sources

## Components

### Data Ingestion
Feeds data from various sources into the database for processing. Data provided via S3(minio) - upload `data_ingestion_files.zip` contents first.

```bash
python -m data_ingestion.main
```

### Interactive Analysis (Marimo)
Main analysis environment with donor behavior modeling, geographic analysis, and ML predictions:

```bash
marimo edit experiment/draw.py
```

### Face Features Extraction
Extracts face features from images and stores them in parquet format:

```bash
cd face_features_app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Maintenance

### Regular Tasks
1. **Update dependencies**: Review and update versions in `pyproject.toml`
2. **Clean virtual environment**: Remove and recreate `.venv` if needed
3. **Run tests**: Ensure all functionality works after changes
4. **Backup data**: Regular database and S3 backups

### Adding New Features
1. Create new modules in appropriate directories (`lib/`, `locations/`, etc.)
2. Add `__init__.py` files for new package directories
3. Update `pyproject.toml` if new dependencies are needed
4. Import cleanly in notebooks: `from lib.new_module import function`