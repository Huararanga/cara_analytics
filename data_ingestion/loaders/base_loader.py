from sqlalchemy import inspect, insert
from sqlalchemy.orm import Session
import pandas as pd

class BaseLoader:
    def __init__(self, model_class, engine):
        self.model_class = model_class
        self.engine = engine

    def _reset_table(self):
        """Drop and recreate the table for this model."""
        insp = inspect(self.engine)
        table = self.model_class.__table__

        if insp.has_table(table.name, schema=table.schema):
            print(f"Dropping existing table: {table}")
            table.drop(self.engine)

        print(f"Creating new table: {table}")
        table.create(self.engine)

    @staticmethod
    def _normalize_records(df: pd.DataFrame):
        """Convert DataFrame to list of dicts with NaT/NaN replaced by None."""

        df = df.copy()

        # Handle datetime columns first
        for col in df.select_dtypes(include=["datetimetz", "datetime64[ns]"]).columns:
            df[col] = df[col].astype("object")   # drop datetime64 dtype
            df[col] = df[col].where(df[col].notna(), None)  # replace NaT/NaN with None

        # Replace NaN/pd.NA in all other columns
        df = df.where(df.notna(), None)

        # Convert to dict and ensure nan values become None
        records = df.to_dict(orient="records")
        
        # Post-process to handle any remaining nan values
        import math
        for record in records:
            for key, value in record.items():
                if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
                    record[key] = None
        
        return records

    def load(self, df, method="bulk_core", chunksize=1000):
        """
        Load DataFrame into DB table with different methods.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to insert.
        method : str
            "to_sql" (default), "bulk_core", "bulk_mappings", "bulk_save".
        chunksize : int
            Number of rows per batch.
        """
        self._reset_table()

        if method == "to_sql":
            # --- Pandas to_sql ---
            print(f"Inserting {len(df)} rows via pandas.to_sql()...")
            df.to_sql(
                name=self.model_class.__tablename__,
                con=self.engine,
                schema=self.model_class.__table__.schema,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=chunksize,
            )

        elif method == "bulk_core":
            # --- SQLAlchemy Core bulk insert ---
            print(f"Inserting {len(df)} rows via bulk_core...")
            with self.engine.begin() as conn:
                for start in range(0, len(df), chunksize):
                    chunk = df.iloc[start:start+chunksize]
                    data = self._normalize_records(chunk)
                    conn.execute(insert(self.model_class), data)
        else:
            raise ValueError(f"Unknown method: {method}")