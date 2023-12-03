from src.services.llm.training import PromptResponseTemplate
from src.services.llm.training.experience_template_generator import WorkExperienceTemplateGenerator
from src.utils import read_json, generate_unique_id, pair_list_items

FOCUS_DATA_ATTRIBUTES = ['certifications', 'education', 'name', 'skills', 'languages', 'summary',
                         'totalYearsExperience', 'profession', 'workExperience']
"""
Critical data attributes from the json output from the resume parsing process with the affinda api
"""


class AffindaTemplateGenerator:
    """
    Fine-tuning prompt response template generator based on the edenai.affinda resume data and information extraction
    """
    def __init__(self, reference: str, resume_data: dict):
        self.data: dict = resume_data
        self.reference = reference

    def generate_templates(self):
        """
        Generates a list of prompt response pairs for fine-tuning
        """
        templates = []
        for key, value in self.data.items():
            if key in FOCUS_DATA_ATTRIBUTES:
                templates.extend(self._get_templates(key, value))
        return templates

    def _get_templates(self, attribute, value):
        if attribute == "certifications":
            return self._get_certifications_templates(value)

        if attribute == "education":
            return self._get_education_templates(value)

        if attribute == "name":
            prompts = ["What is the name of the candidate with application number {} ?".format(self.reference)]
            response = "The candidates/job applicants name is: {}".format(value)
            return list(map(lambda prompt: PromptResponseTemplate(prompt, response), prompts))

        if attribute == "skills":
            return self._get_skills_templates(value)

        if attribute == "languages":
            return self._get_language_templates(value)

        if attribute == "summary":
            prompts = [
                "How can the candidate/job applicants profile be best summarised based on the available resume information ?",
                "From the candidates resume details, write a short summary about the candidate"]
            response = "Based on information fom the candidates' resume, the candidates' information can be summarized as follows: \n {}".format(
                value)
            return list(map(lambda prompt: PromptResponseTemplate(prompt, response), prompts))

        if attribute == "totalYearsExperience":
            prompts = [
                "How many years of work experience does the candidate have ?",
                "From the candidates resume details, how long has the candidate been working for ?"]
            response = "The has work experience of {} years".format(value)
            return list(map(lambda prompt: PromptResponseTemplate(prompt, response), prompts))

        if attribute == "profession":
            prompts = ["What is the candidates profession ?"]
            response = "The has work experience of {} years".format(value)
            return list(map(lambda prompt: PromptResponseTemplate(prompt, response), prompts))

        if attribute == "workExperience":
            return WorkExperienceTemplateGenerator(self.reference, value).generate_templates()

    def _get_certifications_templates(self, value):
        certifications = value or []
        prompts = ["What certifications does the candidate have ?",
                   "Does the candidate have professional certifications ?"]
        responses = []
        if len(certifications) > 0:

            certifications_list_string = "".join(
                ("\n {}".format(value) for value in certifications))
            responses.append(
                "The candidate has a total of {} different certifications. With the details below: {}".format(
                    len(certifications), certifications_list_string))
        responses.append("This candidate has no certifications")
        return list(map(lambda pr: PromptResponseTemplate(pr[0], pr[1]), pair_list_items(prompts, responses)))

    def _get_skills_templates(self, value):
        skills = value or []
        prompts = ["What skills does the candidate have ?",
                   "Does the candidate have professional certifications ?"]
        responses = []
        if len(skills) > 0:
            skills_list_string = "\n".join(
                "{}. {}".format(index + 1, skill['name']) for index, skill in enumerate(skills))
            responses.append(
                "The information from the candidates resume indicates the candidate possesses {} skills with the details below: \n {}".format(len(skills), skills_list_string))
        responses.append("There is no indication of the skills the candidate possesses from the information available")
        return list(map(lambda pr: PromptResponseTemplate(pr[0], pr[1]), pair_list_items(prompts, responses)))

    def _get_language_templates(self, value):
        languages = value or []
        prompts = ["How many languages can the candidate communicate in ?",
                   "Does the candidate speak multiple languages ?"]
        responses = []
        if len(languages) > 0:
            spoken_languages = "\n".join(
                "{}. {}".format(index + 1, language) for index, language in enumerate(languages))
            responses.append(
                "The information from the candidates resume indicates the candidate can speak {} languages, listed below: {}".format(
                    len(languages), spoken_languages))
        responses.append(
            "There is no indication of the spoken languages of the candidate from the available information, however," +
            " it can be assumed the candidate speaks at least one internationally recognized official language")
        return list(map(lambda pr: PromptResponseTemplate(pr[0], pr[1]), pair_list_items(prompts, responses)))

    def _get_education_templates(self, value):
        education_history = value or []
        prompts = ["What educational qualifications does the candidate have ?",
                   "What educational background does the candidate have ?",
                   "What is the candidates educational specialization?",
                   ]
        responses = []
        if len(education_history) > 0:
            education_list_string = []
            for education_data in education_history:
                education_detail = None
                if education_data['grade']['raw'] is not None:
                    education_detail = "The candidate obtained a {} at {} with a grade of {}".format(
                        education_data['accreditation']['education'],
                        education_data['organization'],
                        education_data['grade']['raw'],
                    )
                if education_detail is not None:
                    responses.append(education_detail)
        responses.append("There is no information regarding the candidates educational history from the candidate information available")
        return list(map(lambda pr: PromptResponseTemplate(pr[0], pr[1]), pair_list_items(prompts, responses)))


def generate_fine_tuning_data(resume_json_path: str) -> list:
    """
    Generates a data template for fine-tuning responses to queries on candidate resume based on the affinda engine
    :param resume_json_path: The file path to the json representation of a resume
    :return:
    """
    resume_data = read_json(resume_json_path)
    template_generator = AffindaTemplateGenerator(generate_unique_id(), resume_data[0]['data'])
    return template_generator.generate_templates()



