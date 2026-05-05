from fastapi import FastAPI
from groq import Groq
import os
import psutil
import re

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ----------- Health check -----------
@app.get("/")
def home():
    return {"status": "NovaAI Backend Running"}

# ----------- Chat -----------
@app.get("/chat")
def chat(prompt: str):
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=512
        )

        content = response.choices[0].message.content

        # 🔥 Remove <think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        return {"response": content.strip()}

    except Exception as e:
        return {"error": str(e)}

# ----------- System stats -----------
@app.get("/system")
def system():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent
    }

# ----------- AI Optimization -----------
@app.get("/analyze")
def analyze():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    prompt = f"CPU usage is {cpu}% and RAM usage is {ram}%. Give short optimization advice."

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",   # ✅ fixed model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=256
        )

        content = response.choices[0].message.content
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        return {
            "cpu": cpu,
            "ram": ram,
            "advice": content.strip()
        }

    except Exception as e:
        return {"error": str(e)}
