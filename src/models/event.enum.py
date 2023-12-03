from enum import Enum


class EngineEvent(Enum):
    """
    Enum for the service events in initiaiting a model state update(i.e mode fine-tuning)
    """
    INITIAL_TRAINING = 1,   # This event covers initial training of the model with basic company information
    JOB_ROLE_CREATION = 2,
    JOB_ROLE_UPDATE = 3,
    JOB_AD_CREATION = 4,
    JOB_AD_UPDATE = 5,
    JOB_APPLICATION_ENTRY = 6,
    JOB_HIRE = 7,   # Will not be covered in this version
    JOB_FIRE = 8,   # Will not be covered in this version
