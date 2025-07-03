from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

class OpenAIConnection:
    def __init__(self):
        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be set in environment variables.")
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview"
        )

    def get_client(self):
        return self.client
    
