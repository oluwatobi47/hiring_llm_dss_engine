import traceback

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models.utility_models import ApiResponse
from src.routers import model, inference
from src.services.llm.training import generate_fine_tuning_data
from src.utils import load_app_routes


# Load Environment Variables
load_dotenv('.env')

# Initialize Fast API
app = FastAPI(
    title="Hiring DSS ML Engine",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Load application routes
load_app_routes(app, [
    {
        'prefix': '/model',
        'tags': ['Model Utilities'],
        'router': model.router
    },
    {
        'prefix': '/inference',
        'tags': ['Model Inference'],
        'router': inference.router
    },
    {
        'prefix': '/pipeline',
        'tags': ['Data Pipeline Processing'],
        'router': inference.router
    }
])


@app.get("/")
async def root():
    return {"message": "Welcome to the Hiring LM Based DSS AI Engine"}


@app.get("/generate-tuning-template")
def generate_tuning_data() -> ApiResponse:
    json_resume_path = "/Users/tobialao/Desktop/Software Projects/msc_project/hiring_llm_dss_engine/resources/data/raw" \
                       "-it_support_proffessional.json"
    output = None
    try:
        output = generate_fine_tuning_data(json_resume_path)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(message="Failed to generate tuning data", status="error")

    return ApiResponse(data=output, message="Generated the template successfully", status="success")
