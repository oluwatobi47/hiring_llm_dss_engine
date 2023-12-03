# Imports

import traceback

from fastapi import FastAPI
from dotenv import load_dotenv

from src.parsers.resume import ResumeParser
from src.services.llm.training import generate_fine_tuning_data
from src.utils import read_single_pdf, get_pdf_files
from src.utils.service_simulator import MockJobApplication

# Load Environment Variables
load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the Hiring LM Based DSS"}


@app.get("/generate-tuning-template")
def generate_tuning_data():
    json_resume_path = "/Users/tobialao/Desktop/Software Projects/msc_project/hiring_llm_dss_engine/resources/data/raw" \
                       "-it_support_proffessional.json"
    output = None
    try:
        output = generate_fine_tuning_data(json_resume_path)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return {
            "message": "Failed to generate tuning data"
        }

    return {
        "message": "Generated the template successfully",
        "data": output
    }


# Deprecated
@app.get("/extract-resume-to-json")
async def process_resume_to_json():
    json_resumes = []
    job_applications = MockJobApplication.get_mock_samples()
    job_application: MockJobApplication
    try:
        for job_application in job_applications:
            resume_str = read_single_pdf(job_application.resume_link)
            parser = ResumeParser(resume_str)
            json_resumes.append(parser.get_JSON())
    except Exception as e:
        print(e)
        traceback.print_exc()

    files = get_pdf_files("resources/data/candidate_resumes")
    return {
        'files': files,
        'json_resume': json_resumes
    }
