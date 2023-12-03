from src.services.llm.training import PromptResponseTemplate


class DefaultTemplateGenerator:
    def __init__(self, reference: str, data: dict):
        self.data = data
        self.reference = reference

    def generate_templates(self) -> list:
        templates: list = []
        for key, value in self.data.items():
            context = "Candidate with reference: {} has this attribute {}: {}".format(self.reference, key, value)
            prompt = "With respect to the candidate information, what is the value of this attribute {}".format(key)
            response = "The value of the attribute {} for the job applicant is {}".format(key, value)
            templates.append(PromptResponseTemplate(prompt, response, context))
        return templates


