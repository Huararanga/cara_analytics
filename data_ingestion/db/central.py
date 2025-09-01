from sqlalchemy import create_engine

_engine_cache = {}

def initCentral(DB_URL):
    if "engine" not in _engine_cache:
        print("Creating new MySQL engine...")
        _engine_cache["engine"] = create_engine(DB_URL)

def get_central_engine():
    if "engine" not in _engine_cache:
        raise Exception("initCentral must be called first")
    return _engine_cache["engine"]
