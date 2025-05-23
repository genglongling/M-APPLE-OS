import json
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from dotenv import load_dotenv
from openai import OpenAI
from prompt_templates.data_analysis_template import PROMPT_TEMPLATE, SYSTEM_PROMPT
from skills.skill import Skill

load_dotenv()


class AnalyzeData(Skill):
    def __init__(self):
        super().__init__(self.NAME, self.ANALYZE_DATA_FUNCTION_DICT, self.data_analyzer)

    NAME = "data_analyzer"
    ANALYZE_DATA_FUNCTION_DICT = {
        "type": "function",
        "function": {
            "name": NAME,
            "description": "Provides insights, trends, or analysis based on the data and prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The data to analyze."},
                    "prompt": {
                        "type": "string",
                        "description": "The original user prompt that the data is based on.",
                    },
                },
                "required": ["data", "prompt"],
            },
        },
    }

    # Define the data analysis tool function
    def data_analyzer(self, args):
        """Provides insights, trends, or analysis based on the data and prompt.

        Args:
            args (dict): A dictionary containing the data to analyze and
            the original user prompt that the data is based on.

        Returns:
            str: The analysis result.
        """

        if isinstance(args, dict) and "prompt" in args and "data" in args:
            prompt = args["prompt"]
            data = args["data"]
        elif isinstance(args, str):
            try:
                args = json.loads(args)
                prompt = args["prompt"].strip()
                data = args["data"].strip()
            except ValueError:
                return "Invalid input: expected a dictionary or a string."
        else:
            return "Invalid input: expected a dictionary or a string."

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(PROMPT=prompt, DATA=data),
                },
            ],
        )
        return response.choices[0].message.content
