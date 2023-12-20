import datetime
import random
import uuid
from dataclasses import dataclass, asdict
from typing import Optional

from src.models.data_models import JobDescription, JobPost, JobApplication, Company
from src.utils import save_json, get_files, get_file_name, read_json


@dataclass
class DataPool:
    job_descriptions: list[JobDescription]
    job_posts: list[JobPost]
    job_applications: list[JobApplication]
    company_info: Optional[Company] = None


class DataSynthesizer:
    _company_info = {
        'id': '845bfe0e-538d-4c38-897e-3b30b0d3458b',
        'address': 'Mountain View, California, USA',
        'company_culture_statement': 'A problem isn''t truly solved until it''s solved for all. Googlers build products that help create opportunities for everyone, whether down the street or across the globe. Bring your insight, imagination and a healthy disregard for the impossible. Bring everything that makes you unique. Together, we can build for everyone',
        'date_founded': str(datetime.date.fromtimestamp(904867200)),
        'mission_statement': 'To organize the world''s information and make it universally accessible and useful',
        'name': 'Test Google',
        'vision': 'To provide access to the world''s information in one click'
    }

    def __init__(self, datapool_path: str, base_test_case_path: str, test_case_config: dict):
        self.datapool_path = datapool_path
        self.base_test_case_path = base_test_case_path
        self.test_case_config = test_case_config

    def _load_data_pool(self) -> DataPool:
        base_test_case_pool: dict | DataPool = read_json(self.base_test_case_path)

        # TODO (Future works): Parameterize max variables of 20, 20, and 100
        max_jds = 20 - len(base_test_case_pool['job_descriptions'])
        max_posts = 20 - len(base_test_case_pool['job_posts'])
        max_applications = 100 - len(base_test_case_pool['job_applications'])

        job_descriptions = self._create_job_descriptions(self._company_info['id'], limit=max_jds)
        job_descriptions = list(
            map(lambda x: JobDescription(**x), base_test_case_pool['job_descriptions'])) + job_descriptions

        job_posts = self._create_job_posts(job_descriptions, limit=max_posts)
        job_posts = list(map(lambda x: JobPost(**x), base_test_case_pool['job_posts'])) + job_posts

        job_applications = self._create_job_applications(job_posts, limit=max_applications)
        job_applications = list(
            map(lambda x: JobApplication(**x), base_test_case_pool['job_applications'])) + job_applications

        data = DataPool(
            job_applications=job_applications,
            job_posts=job_posts,
            job_descriptions=job_descriptions,
            company_info=Company(**self._company_info)
        )

        json_data = {
            "company_info": self._company_info,
            "job_applications": list(map(lambda x: asdict(x), job_applications)),
            "job_posts": list(map(lambda x: asdict(x), job_posts)),
            "job_descriptions": list(map(lambda x: asdict(x), job_descriptions))
        }

        try:
            save_json(json_data, self.datapool_path)
        except Exception as e:
            print(e)

        return data
        # load base test cases into memory
        # Based on dataset files
        # Create job

    def _create_job_descriptions(self, company_id, limit=20) -> list[JobDescription]:
        desc_doc_path = "/Users/tobialao/Desktop/Software_Projects/msc_project/hiring_llm_dss_engine/resources/dataset/job_description"
        documents = get_files(desc_doc_path, extension_matcher="*.docx")
        job_descriptions = []
        file_limit = limit if limit <= len(documents) else len(documents)
        for index in range(file_limit):
            job_descriptions.append(
                JobDescription(
                    id=str(uuid.uuid4()),
                    company_id=company_id,
                    job_title=get_file_name(documents[index], f"Job Description {index}"),
                    job_description_file=documents[index]
                )
            )
        return job_descriptions

    def _create_job_posts(self, job_description_pool: list[JobDescription], limit=20) -> list:
        job_posts = []
        random_indices = random.sample(range(len(job_description_pool)), len(job_description_pool))
        range_limit = limit if limit <= len(random_indices) else len(random_indices)
        salary_ranges = ["$60,000 - $80,000", "$80,000 - $100,000", "$100,000 - $120,000", "$120,000 - $140,000",
                         "$140,000+"]
        for index in range(range_limit):
            job_posts.append(
                JobPost(
                    id=str(uuid.uuid4()),
                    created_date=str(datetime.datetime.now()),
                    title=job_description_pool[index].job_title,
                    job_description_id=job_description_pool[index].id,
                    active=True if index % 3 == 1 else False,
                    salary_range=random.choice(salary_ranges)
                )
            )
        return job_posts

    def _create_job_applications(self, job_post_pool: list[JobPost], limit=20) -> list:
        job_applications = []
        resume_path = "/Users/tobialao/Desktop/Software_Projects/msc_project/hiring_llm_dss_engine/resources/dataset/resume"
        documents = get_files(resume_path, extension_matcher="*.pdf")
        random_doc_indices = random.sample(range(len(documents)), min(limit, len(documents)))
        jp_pool_range = range(len(job_post_pool))
        for index in range(len(random_doc_indices)):
            job_applications.append(
                JobApplication(
                    id=str(uuid.uuid4()),
                    created_date=str(datetime.datetime.now()),
                    candidate_name=f"Job Applicant {index}",
                    candidate_email=f"job_applicant_{index}@hr_dss_email.com",
                    job_post_id=job_post_pool[random.choice(jp_pool_range)].id,
                    resume_link=documents[index],
                    active=True if index % 3 == 1 else False,
                )
            )
        return job_applications

    def get_data(self, test_case_code: str) -> DataPool:
        datapool = None
        try:
            datapool_dict = read_json(self.datapool_path)
            datapool = DataPool(
                company_info=Company(**datapool_dict['company_info']),
                job_descriptions=list(map(lambda x: JobDescription(**x), datapool_dict['job_descriptions'])),
                job_posts=list(map(lambda x: JobPost(**x), datapool_dict['job_posts'])),
                job_applications=list(map(lambda x: JobApplication(**x), datapool_dict['job_applications']))
            )
        except FileNotFoundError as e:
            print(e)
        finally:
            if datapool is None:
                datapool = self._load_data_pool()

        config = self.test_case_config[test_case_code]
        return DataPool(
            company_info=datapool.company_info,
            job_descriptions=datapool.job_descriptions[:config['job_descriptions']],
            job_posts=datapool.job_posts[:config['job_posts']],
            job_applications=datapool.job_applications[:config['job_applications']]
        )
