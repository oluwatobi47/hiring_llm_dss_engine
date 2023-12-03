from src.services.llm.training import PromptResponseTemplate


class SingleAttributeTemplateGenerator:
    def __init__(self, reference: str, attribute: str, value: str, context=""):
        self.attribute_name = attribute
        self.attribute_value = value,
        self.context = context
        self.reference = reference

    def generate_templates(self):
        context = self.context if self.context is not None and len(self.context) > 0 \
            else "Candidate with reference: {} has this attribute {}: {}".format(
            self.reference, self.attribute_name, self.attribute_value) if self.attribute_value else ""
        prompt = "With respect to the candidate information provided, what is the value of this attribute {} ?".format(
            self.attribute_name)
        response = "The value of the attribute {} for the job applicant is {}".format(
            self.attribute_name, self.attribute_value) if self.attribute_value is not None \
            else "This information about the job applicant is not available"
        return PromptResponseTemplate(prompt, response, context)
