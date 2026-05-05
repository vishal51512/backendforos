from fastapi import FastAPI
from groq import Groq
import os
import psutil

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Health check
@app.get("/")
def home():
    return {"status": "NovaAI Backend Running"}

# Chat endpoint
@app.get("/chat")
def chat(prompt: str):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"response": response.choices[0].message.content}

# System stats
@app.get("/system")
def system():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent
    }

# AI optimization
@app.get("/analyze")
def analyze():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    prompt = f"""
    CPU usage is {cpu}% and RAM usage is {ram}%.
    Give short optimization advice.
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "cpu": cpu,
        "ram": ram,
        "advice": response.choices[0].message.content
    }
