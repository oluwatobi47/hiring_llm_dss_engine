class JobApplication:
    """
    When a job application data is received, the pipeline processes the candidate data
    The pipeline engine can be a fully fledged service which detects and extracts all new attributes
    from json data sent from client system
    """

    def __init__(self, data):
        self.ref = ""  # Job application reference
        self.job_ad_ref = ""  # Job role/post/ad reference
        self.candidate_data = {}  # Standard model fed to data pipeline based on existing system data structure
        self.resume_link = ""  # Link to candidates resume document
        self.cover_page_link = ""  # Not implementing
