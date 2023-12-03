import json
import requests

API_KEY = ""
TEMPLATE_KEY = ""

class EdenAIResumeParser:
    def __init__(self, resume_file_url: str):
        self.url = resume_file_url

    async def parseResume(self) -> dict:
        headers = {"Authorization": "Bearer {}".format(API_KEY)}
        url = "https://api.edenai.run/v2/ocr/resume_parser"
        json_payload = {
            "show_original_response": False,
            "fallback_providers": "",
            "providers": "affinda",
            "file_url": self.url
        }
        response = requests.post(url, json=json_payload, headers=headers)

        result = json.loads(response.text)
        print(result["affinda"]["extracted_data"])
        return result
