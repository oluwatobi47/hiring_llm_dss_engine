import traceback

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from src.models.utility_models import ApiResponse
from src.services.llm.training import generate_fine_tuning_data
from src.routers import model
# Load Environment Variables
load_dotenv()

# Initialize Fast API
app = FastAPI()
ALLOWED_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(model.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Hiring LM Based DSS AI Engine"}


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
        return ApiResponse(message="Failed to generate tuning data", status="error")

    return ApiResponse(data=output, message="Generated the template successfully", status="success")
