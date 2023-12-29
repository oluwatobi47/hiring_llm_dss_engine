import json
import os

from src.services.data_pipeline.data_ingestion_service import DataIngestionService
from src.services.evaluation.data_synthesizer import DataSynthesizer, DataPool
from src.services.evaluation.metrics_data_persistence import MetricsDataService, EvaluationBatch, EvaluationResults
from src.services.inference import InferenceService
from src.utils import Benchmarker, read_json


class SimulationService:

    def __init__(self, inference_service: InferenceService,
                 data_ingestion_service: DataIngestionService,
                 metrics_service: MetricsDataService):
        # Synthesize datapool if it does not exist
        self.inference_service = inference_service
        self._datapool = None
        self.metrics_service = metrics_service
        self.data_ingestion_service = data_ingestion_service

    def get_test_cases(self):
        test_case_config_path = "{}/test_case_config.json".format(os.getenv("EVAL_DATA_PATH"))
        data = read_json(test_case_config_path)
        test_cases = []
        if isinstance(data, dict):
            for key in data:
                test_cases.append(data[key])
        return test_cases

    def _write_data_to_stores(self, datapool: DataPool):
        for description in datapool.job_descriptions:
            self.data_ingestion_service.process_job_description_create(description)
        for application in datapool.job_applications:
            self.data_ingestion_service.process_job_application_create(application)
        for post in datapool.job_posts:
            self.data_ingestion_service.process_job_post_create(post)
        if datapool.company_info is not None:
            self.data_ingestion_service.process_company_info_create(datapool.company_info)

    def _clean_data(self):
        """Clears vector store of all collection data"""
        self.data_ingestion_service.get_client().reset()
        self.data_ingestion_service.refresh_datasource()

        # Collections in vector DB
        # collections = ["company", "resume", "job_post", "job_description"]
        # for collection_name in collections:
        #     if collection_name in map(lambda x: x.name, self.data_ingestion_service.get_client().list_collections()):
        #         collection = self.data_ingestion_service.get_client().get_collection(collection_name)
        #         if len(collection.get()["ids"]) > 0:
        #             collection.delete(collection.get()["ids"])
        #             self.data_ingestion_service.get_client().clear_system_cache()

    def run_simulation(self, simulation_batch: EvaluationBatch):
        questions_path = "{}/truthful_qa_questions.json".format(os.getenv("EVAL_DATA_PATH"))
        datapool_path = "{}/evaluation_data_pool.json".format(os.getenv("EVAL_DATA_PATH"))
        base_test_case_data_path = "{}/base_test_case_data.json".format(os.getenv("EVAL_DATA_PATH"))
        test_case_config_path = "{}/test_case_config.json".format(os.getenv("EVAL_DATA_PATH"))

        # Load testcase configuration and data
        test_case_config = read_json(test_case_config_path)
        questions: list = read_json(questions_path)
        data_synthesizer = DataSynthesizer(
            datapool_path=datapool_path,
            base_test_case_path=base_test_case_data_path,
            test_case_config=test_case_config)
        dataset: DataPool = data_synthesizer.get_data(simulation_batch.test_case_code)

        # Clean data store of current data
        self._clean_data()

        # Write data to data stores
        self._write_data_to_stores(dataset)

        # Re-construct query_engine
        self.inference_service.refresh_datasource()

        benchmarker = Benchmarker()
        benchmarker.start()
        process_bm = Benchmarker()

        # Execute response to queries
        for question in questions:
            error = None
            response = None
            try:
                process_bm.start()
                response = self.inference_service.generate_response(question['question'])
            except Exception as e:
                error = str(e)
            finally:
                process_bm.end()
                prompt_token = self.inference_service.get_token_count(question['question'])
                response_token = self.inference_service.get_token_count(response)

                extra_info = {
                    "prompt_token": prompt_token,
                    "response_token": response_token
                }
                metadata = json.dumps(extra_info)
                metric_data = EvaluationResults(
                    batch_id=simulation_batch.id,
                    question_id=question["number"],
                    response=response,
                    error=error,
                    execution_time=process_bm.get_execution_time(),
                    process_metadata=metadata
                )
                self.metrics_service.get_db_session().add(metric_data)
                self.metrics_service.get_db_session().commit()

        benchmarker.end()
        simulation_batch.execution_time = benchmarker.get_execution_time()
        simulation_batch.completed = True
        self.metrics_service.get_db_session().commit()
        self.metrics_service.get_db_session().close()
