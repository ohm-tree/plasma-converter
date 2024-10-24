from openai import OpenAI
import os

client = OpenAI(
    api_key = os.getenv('OPENAIKEY')
)
    
models = client.models.list()

print(models)