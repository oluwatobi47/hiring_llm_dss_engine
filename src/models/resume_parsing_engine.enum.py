from enum import Enum


class ResumeParsingEngine(Enum):
    """
    Enum for supported resume parsing engines
    """
    EDEN_AI_AFFINDA = "edenai.affinda",
    PROJECT_CUSTOM = "custom",

