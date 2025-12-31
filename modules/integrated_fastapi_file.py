import os
import re
import json
import uuid
import asyncio
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import google.generativeai as genai
from openai import AsyncOpenAI

# =====================================================
# 1️⃣ Load environment variables
# =====================================================
load_dotenv()

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY or not GEMINI_MODEL:
    raise ValueError("❌ Missing GEMINI_API_KEY or GEMINI_MODEL in .env")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

# =====================================================
# 2️⃣ Configure APIs
# =====================================================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# 3️⃣ Create FastAPI App
# =====================================================
app = FastAPI()

# CORS for Flutter or React apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder for generated TTS audio
os.makedirs("tts_outputs", exist_ok=True)

# =====================================================
# 4️⃣ Helper: Safe JSON Parsing for Gemini
# =====================================================
def parse_gemini_json(response_text: str):
    cleaned = re.sub(r'```(?:json)?', '', response_text, flags=re.I)
    cleaned = cleaned.strip('`\n ')
    cleaned = re.sub(r'""(.*)""', r'"\1"', cleaned)

    try:
        return json.loads(cleaned)
    except:
        match = re.search(r'\{.*\}', cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

    return {"text": cleaned, "emotion": "unknown"}

# =====================================================
# 5️⃣ Gemini Emotion Classifier
# =====================================================
def classify_emotion_sinhala(text):
    prompt = f"""
You are an advanced Sinhala emotion and voice-style classifier.

Task:
1. Detect the emotion in this Sinhala or tamil sentence.
2. Provide voice-style parameters for TTS.

Emotion classes:
happy, sad, fear, anger, surprise, neutral.

Return ONLY valid JSON:
{{
  "text": "{text}",
  "emotion": "<one_of:[happy,sad,fear,anger,surprise,neutral]>",
  "voice_affect": "<voice style>",
  "tone": "<tone details>",
  "pacing": "<pacing style>",
  "emotion_description": "<explanation>",
  "personality": "<personality traits>",
  "pauses": "<pause style>"
}}

Analyze and output JSON only.
Sentence: "{text}"
"""
    try:
        response = gemini_model.generate_content(prompt)
        return parse_gemini_json(response.text)
    except Exception as e:
        print("Gemini Error:", e)
        return {"text": text, "emotion": "error"}

# =====================================================
# 6️⃣ TTS Generation with OpenAI
# =====================================================
async def generate_tts_audio(text, instructions):
    file_id = f"{uuid.uuid4()}.wav"
    output_path = os.path.join("tts_outputs", file_id)

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions=instructions,
        response_format="wav",
    ) as response:

        # Save streamed audio chunks
        with open(output_path, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)

    return file_id

# =====================================================
# 7️⃣ API Endpoint (Gemini → OpenAI TTS)
# =====================================================
@app.post("/classify-emotion-tts/")
async def classify_emotion_tts(text: str = Form(...)):
    # 1️⃣ Gemini: Emotion Classification
    gemini_result = classify_emotion_sinhala(text)

    # 2️⃣ Prepare TTS Instructions
    tts_instructions = (
        f"Voice: {gemini_result.get('voice_affect','')}\n"
        f"Tone: {gemini_result.get('tone','')}\n"
        f"Pacing: {gemini_result.get('pacing','')}\n"
        f"Personality: {gemini_result.get('personality','')}\n"
        f"Pauses: {gemini_result.get('pauses','')}\n"
        f"Emotion Description: {gemini_result.get('emotion_description','')}"
    )

    # 3️⃣ OpenAI TTS: Generate Audio
    try:
        audio_filename = await generate_tts_audio(gemini_result["text"], tts_instructions)
        audio_url = f"http://localhost:8000/audio/{audio_filename}"
    except Exception as e:
        print("TTS Error:", e)
        audio_url = None

    # 4️⃣ Return Combined Result
    return JSONResponse({
        "success": True,
        "gemini_output": gemini_result,
        "audio_url": audio_url
    })

# =====================================================
# 8️⃣ Audio Serving Route
# =====================================================
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = os.path.join("tts_outputs", filename)

    if not os.path.exists(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(filepath, media_type="audio/wav")

# =====================================================
# 9️⃣ Run Server
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("integrated_fastapi_file:app", host="0.0.0.0", port=8000, reload=True)
