import datetime
import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Column, String, DateTime, Engine, Integer, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session, declared_attr

from src.services.data import load_db_client
from src.utils import read_json

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
    qa_accepted = Column(Boolean, default=False)


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

    def get_batch_evaluation_results(self, batch_ids: list[int]):
        batch_list: list = self._session.query(EvaluationBatch).filter(
            EvaluationBatch.id.in_(batch_ids)).all()
        result = []
        for data in batch_list:
            batch_data = vars(data)
            batch_data['items'] = self._get_evaluation_results(data.id)
            result.append(batch_data)
        return result

    def update_qa_value(self, batch_result_id: int, value: bool):
        result: EvaluationResults = self._session.query(EvaluationResults).get({
            'id': batch_result_id
        })
        result.qa_accepted = value
        self._session.commit()

    def _get_evaluation_results(self, batch_id: int):
        questions_path = "{}/truthful_qa_questions.json".format(os.getenv("EVAL_DATA_PATH"))
        questions: list = read_json(questions_path)
        question_map = {str(item["number"]): item for item in questions}
        result_list = self._session.query(EvaluationResults).filter_by(batch_id=batch_id).all()

        def add_question_info(obj: EvaluationResults):
            data = vars(obj)
            data["question"] = question_map[obj.question_id]['question']
            data["hint"] = question_map[obj.question_id]['hint']
            data["expected_response"] = question_map[obj.question_id]['response']
            return data

        return list(map(add_question_info, result_list))


Base.metadata.create_all(db_engine)
