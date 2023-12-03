class MockCandidateData:
    def __init__(self, data):
        self.ref = data.ref
        self.name = data.name
        self.email = data.email
        self.address = data.address
        self.linkedInUrl = data.linkedInUrl
        self.website = data.website

# TODO: Create a proper candidate data generator which would
#  generate candidate data based on available resumes in specified directory

class MockJobApplication:
    def __init__(self, job_applicant_name: str, application_date, resume_link: str):
        self.resume_link = resume_link
        self.application_date = application_date
        self.applicant_name = applicant_name
        self.job_post_ref

    @staticmethod
    def get_mock_samples() -> list:
        return [
            MockJobApplication("Candidate 1", "01-11-2023", "resources/data/candidate_resumes/accountant_1.pdf"),
            MockJobApplication("Candidate 2", "01-11-2023", "resources/data/candidate_resumes/it_professional_1.pdf"),
        ]

