class PromptResponseTemplate:
    def __init__(self, prompt, response, context=""):
        self.prompt = prompt
        self.response = response
        self.context = context

    def get_prompt(self, add_context=True):
        if add_context and self.context is not None:
            return "Given this context: \n {context} \n\n Prompt/Question: \n {prompt}" \
                .format(context=self.context, prompt=self.prompt)
        else:
            return self.prompt
