import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

response = requests.post(
    OLLAMA_URL,
    json={
        "model": "qwen2.5:7b",
        "prompt": "திருட்டு என்றால் என்ன? எளிய வார்த்தைகளில் விளக்குங்கள்.",
        "stream": False
    }
)

print(response.json()["response"])