from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData

SCHEMA = "external"

Base = declarative_base(metadata=MetaData(schema=SCHEMA))

