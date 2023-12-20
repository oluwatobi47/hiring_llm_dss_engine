import datetime
import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Column, String, DateTime, Engine, Integer, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session, declared_attr

from src.services.data import load_db_client

load_dotenv(find_dotenv('.env'))

Base = declarative_base()
db_engine: Engine = load_db_client(os.getenv('DB_URI'))
LocalSession = sessionmaker(autoflush=False, bind=db_engine)


class BaseModel:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_date = Column(DateTime, default=datetime.datetime.now())
    error = Column(String)
    execution_time = Column(String)
    process_metadata = Column(String)


# Data Models
class EvaluationBatch(BaseModel, Base):
    batch_ref = Column(String)
    summary = Column(String)
    test_case_code = Column(String)
    completed = Column(Boolean, default=False)


class EvaluationResults(BaseModel, Base):
    batch_id = Column(Integer)
    question_id = Column(String)
    response = Column(String)


class OperationMetrics(BaseModel, Base):
    operation = Column(String)


class MetricsDataService:
    _session: Session

    def __init__(self):
        self._session = LocalSession()

    def get_db_session(self):
        return self._session

    def clean_db(self):
        self._session.query(EvaluationBatch).delete()
        self._session.query(EvaluationResults).delete()
        self._session.query(OperationMetrics).delete()

    def find_op_metric_by_id(self, metric_id: int):
        return self._session.query(OperationMetrics).get(metric_id)

    def get_op_metrics(self):
        return self._session.query(OperationMetrics).all()

    def get_evaluation_batches(self):
        return self._session.query(EvaluationBatch).all()

    def get_evaluation_results(self, batch_id: int):
        return self._session.query(EvaluationResults).filter_by(batch_id=batch_id)


Base.metadata.create_all(db_engine)
