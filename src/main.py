from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import model_training_router, model_utility_router, data_pipeline_router, evaluation_router, inference_router
from src.utils import load_app_routes

# Load Environment Variables
load_dotenv('.env')

# Initialize Fast API
app = FastAPI(
    title="Hiring DSS ML Engine",
)

# Load middlewares/interceptors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load application API endpoint routes
load_app_routes(app, [
    {
        'prefix': '/model',
        'tags': ['Model Utilities: Background operations (Download/Upload)'],
        'router': model_utility_router
    },
    {
        'prefix': '/inference',
        'tags': ['Model Inference generation on data sources'],
        'router': inference_router
    },
    {
        'prefix': '/pipeline',
        'tags': ['Data Pipeline Processing'],
        'router': data_pipeline_router
    },
    {
        'prefix': '/train',
        'tags': ['Model Fine-tuning/Training'],
        'router': model_training_router
    },
    {
        'prefix': '/evaluation',
        'tags': ['Model Evaluation & Metrics'],
        'router': evaluation_router
    }
])


# Entry/Root API Endpoint Resource
@app.get("/")
async def root():
    return {"message": "Welcome to the Hiring LM Based DSS AI Engine, Access the docs on /docs"}
