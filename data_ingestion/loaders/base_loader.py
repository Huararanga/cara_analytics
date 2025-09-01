from sqlalchemy import inspect

class BaseLoader:
    def __init__(self, model_class, engine):
        self.model_class = model_class
        self.engine = engine

    def load(self, df):
        # Drop table if it exists
        if inspect(self.engine).has_table(self.model_class.__tablename__, schema=self.model_class.__table__.schema):
            print(f"Dropping existing table: {self.model_class.__table__}")
            self.model_class.__table__.drop(self.engine)

        # Recreate table
        self.model_class.__table__.create(self.engine)

        # Load data
        print(f"Creating new data: {self.model_class.__table__}")
        df.to_sql(
            name=self.model_class.__tablename__,
            con=self.engine,
            schema=self.model_class.__table__.schema,
            if_exists='append',
            index=False,
            method='multi'
        )
