from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from text_processor import extract_headings_and_groups
from state_reader import load_headlines
from rag_engine import ArticleRAG
from web_fallback import WebFallbackService
from voice_io import speak, speech_to_text

import shutil
import os

app = FastAPI()

ARTICLE_FILE = f"D:\\final Year project\\Research_backend\\RAG\\test2.txt"
LLM_MODEL_PATH = f"D:\\final Year project\\Research_backend\\LLM_Talk_Back\\models\\tamil-llama-7b-v0.1-q8_0.gguf"

rag_engine = None
web_fallback = WebFallbackService(model_path=LLM_MODEL_PATH)


# -------------------------
# 1️⃣ Upload article text
# -------------------------
@app.post("/upload_article")
def upload_article(file: UploadFile = File(...)):
    global rag_engine

    with open(ARTICLE_FILE, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(ARTICLE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    results = extract_headings_and_groups(text)

    # initialize RAG with new article
    rag_engine = ArticleRAG(
        text_file_path=ARTICLE_FILE,
        model_path=LLM_MODEL_PATH
    )

    # Announce headlines immediately
    for item in results:
        speak(item["headline"])

    return {"status": "article processed"}


# -------------------------
# 2️⃣ Replay headlines
# -------------------------
@app.get("/replay_headlines")
def replay_headlines():
    headlines = load_headlines()
    for h in headlines:
        speak(h)
    return {"status": "replayed"}


# -------------------------
# 3️⃣ Voice question handling
# -------------------------
@app.post("/ask_voice")
@app.post("/ask_voice")
# -------------------------
# 3️⃣ Voice question handling
# -------------------------
@app.post("/ask_voice")
def ask_voice():
    global rag_engine

    question = speech_to_text()

    if not question:
        return JSONResponse(
            content={"error": "No speech detected"},
            status_code=400
        )

    print("🎤 USER QUESTION:", question)

    # Special command
    if "மீண்டும் தலைப்புகளை" in question:
        headlines = load_headlines()
        for h in headlines:
            speak(h)
        return {"status": "headlines replayed"}

    # ---- Article RAG ----
    answer = rag_engine.answer(question)

    # If answer NOT in document → go web directly
    if "எனக்கு தெரியவில்லை" in answer or "இணையத்தில் தேட" in answer:
        speak("இந்த தகவல் கொடுக்கப்பட்ட ஆவணத்தில் இல்லை. இணையத்தில் தேடுகிறேன், தயவு செய்து காத்திருக்கவும்.")

        print("🌐 DIRECT WEB SEARCH TRIGGERED")
        web_answer = web_fallback.answer(question)

        speak(web_answer)
        return {"source": "web", "question": question}

    # ---- Normal RAG answer ----
    speak(answer)
    return {"source": "article", "question": question}

