from src.services.llm.training import PromptResponseTemplate
from src.utils import pair_list_items
# from transformers import AutoModelForCausalLM


class WorkExperienceTemplateGenerator:
    """
    Given a list of work experiences

    """
    def __init__(self, reference, data: dict):
        """
        :param reference: The unique reference to the owning document holder
        :param data: The list of work experience information based on the data struction from the affinda engine
        """
        self.reference = reference
        self.experiences = data or []

    def generate_templates(self) -> list:
        # AutoModelForCausalLM.from_config(pre)
        prompts = ["In how many different roles or capacity as the applicant worked before in the past ?",
                   "Where has the candidate worked before ?",
                   "Tell me about the candidates work history",
                   "What organizations has the candidate worked with before ?"
                   ]

        responses = ["The candidate has no prior work history or experience",
                     "The candidate has worked in five different roles before"
                     ]
        if len(self.experiences) > 0:
            experience_list_string = []
            for experience in self.experiences:
                detail = "\n Worked as a {jobTitle} at {organization} from {startDate} {endDateText}".format(
                    jobTitle=experience['jobTitle'],
                    organization=experience['organization'],
                    startDate=experience['dates']['startDate'],
                    endDateText="till date" if experience['dates']['isCurrent'] else "to {}".format(experience['dates']['endDate']),
                )
                if experience['jobDescription'] and len(experience['jobDescription']) > 0:
                    detail + "Job Functions: \n {jobDesc}".format(jobDesc=experience['jobDescription'])
                experience_list_string.append(detail)
            responses.append("Details of the applicants work history are as follows: " + "".join(experience_list_string))
        return list(map(lambda pr: PromptResponseTemplate(pr[0], pr[1]), pair_list_items(prompts, responses)))

