from sqlalchemy import create_engine

_engine_cache = {}

def initPostgres(DB_URL):
    if "engine" not in _engine_cache:
        print("Creating new engine...")
        _engine_cache["engine"] = create_engine(DB_URL)

def get_engine():
    if "engine" not in _engine_cache:
        raise Exception("initPostgres must be called first")
    return _engine_cache["engine"]
