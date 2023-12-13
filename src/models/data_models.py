from dataclasses import dataclass


@dataclass
class Company:
    """ Class for keeping track of basic company information """
    id: str
    name: str
    date_founded: str
    mission_statement: str
    vision: str
    company_culture_statement: str
    address: str


@dataclass
class JobDescription:
    """ Class for keeping track of Job description information defined for a specific company"""
    id: str
    company_id: str  # Not required for this model, a saas/paas solution may require this
    job_title: str
    job_description_file: str


@dataclass
class JobPost:
    """ Class for keeping track of Job description information defined for jobs"""
    id: str
    created_date: str
    title: str
    job_description_id: str
    active: bool
    salary_range: str


@dataclass
class JobApplication:
    """ Class for keeping track of Job description information defined for jobs"""
    id: str
    candidate_name: str
    candidate_email: str
    created_date: str
    job_post_id: str
    resume_link: str
    active: bool
