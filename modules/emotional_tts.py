###############################################################
#  ONE FASTAPI BACKEND FOR SINHALA + TAMIL EMOTION + TTS
###############################################################

import os
import re
import json
import emoji
import torch
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# ============================================
# LOAD ENV
# ============================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# COMMON CLEAN TEXT FUNCTION
# ============================================
def clean_text(text: str, lang: str):
    text = emoji.demojize(str(text))
    text = re.sub(r"http\S+|www\S+", "", text)

    if lang == "si":  # Sinhala Unicode range
        text = re.sub(r"[^0-9A-Za-z\u0D80-\u0DFF.,!?\s]", " ", text)
    else:  # Tamil Unicode range
        text = re.sub(r"[^0-9A-Za-z\u0B80-\u0BFF.,!?\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# ================================================================
# 1️⃣ SINHALA MODEL + META
# ================================================================
SINHALA_MODEL_DIR = "sinbert_sinhala_best"
SINHALA_MAX_LEN = 96
SINHALA_TTS_DIR = "tts_outputs_sinhala"
os.makedirs(SINHALA_TTS_DIR, exist_ok=True)

print("📌 Loading Sinhala model...")
si_tokenizer = AutoTokenizer.from_pretrained(SINHALA_MODEL_DIR)
si_model = AutoModelForSequenceClassification.from_pretrained(SINHALA_MODEL_DIR).to(DEVICE)
si_model.eval()

with open(os.path.join(SINHALA_MODEL_DIR, "id2label.json"), "r", encoding="utf-8") as f:
    si_id2label = json.load(f)

# ================================================================
# IMPROVED SINHALA EMOTION META
# ================================================================
SI_EMOTION_META = {
    "happy": {
        "voice_affect": "Warm, cheerful, bright emotional color with genuine joy and a natural smile in the voice",
        "tone": "Lively, friendly, uplifting tone with smooth expressive energy",
        "pacing": "Energetic, rhythmic pacing with natural flow",
        "emotion_description": "Expressing happiness, excitement, comfort, and positive feelings",
        "personality": "Kind, optimistic, joyful, and emotionally open",
        "pauses": "Short, soft pauses that feel playful and natural"
    },
    "sad": {
        "voice_affect": "Low, soft tone with a gentle, heavy-hearted emotional color",
        "tone": "Muted, reflective, slightly trembling tone carrying emotional weight",
        "pacing": "Slow pacing with long emotional pauses and softened delivery",
        "emotion_description": "Deep sadness, disappointment, grief, or emotional pain",
        "personality": "Sensitive, calm, introspective, emotionally delicate",
        "pauses": "Long, deep pauses showing heaviness and emotional reflection"
    },
    "fear": {
        "voice_affect": "Shaky, tense emotional color with audible nervousness",
        "tone": "Hesitant, unstable tone with anxious fluctuations",
        "pacing": "Uneven pacing with sudden short pauses showing tension",
        "emotion_description": "Fear, stress, uncertainty, or nervousness",
        "personality": "Alert, worried, cautious, easily startled",
        "pauses": "Irregular, broken pauses that express fear or hesitation"
    },
    "anger": {
        "voice_affect": "Strong, forceful emotional color with heated energy",
        "tone": "Sharp, firm, intense tone with clear irritation",
        "pacing": "Fast, pressured, and forceful speaking rhythm",
        "emotion_description": "Anger, frustration, conflict, or strong disagreement",
        "personality": "Bold, assertive, direct, and intense",
        "pauses": "Short tight pauses that emphasize strong emotions"
    },
    "surprise": {
        "voice_affect": "Excited, bright emotional color with high alertness",
        "tone": "High-pitched and expressive tone",
        "pacing": "Fast bursts with rising pitch patterns",
        "emotion_description": "Unexpected shock, amazement, confusion, or discovery",
        "personality": "Curious, expressive, reactive",
        "pauses": "Quick dramatic pauses that highlight surprise"
    },
    "neutral": {
        "voice_affect": "Calm, balanced emotional color with steady clarity",
        "tone": "Natural, clear, and even tone without emotional bias",
        "pacing": "Smooth, steady pacing with clear articulation",
        "emotion_description": "Neutral and objective with no emotional load",
        "personality": "Professional, calm, composed",
        "pauses": "Natural pauses with consistent timing"
    }
}


# ================================================================
# 2️⃣ TAMIL MODEL + META
# ================================================================
TAMIL_MODEL_DIR = "best_emotion_model"
TAMIL_MAX_LEN = 128
TAMIL_TTS_DIR = "tts_outputs"
os.makedirs(TAMIL_TTS_DIR, exist_ok=True)

print("📌 Loading Tamil model...")
ta_tokenizer = AutoTokenizer.from_pretrained(TAMIL_MODEL_DIR)
ta_model = AutoModelForSequenceClassification.from_pretrained(TAMIL_MODEL_DIR).to(DEVICE)
ta_model.eval()

ta_id2label = ta_model.config.id2label

# ================================================================
# IMPROVED TAMIL EMOTION META
# ================================================================
TA_EMOTION_META = {
    "happy": {
        "voice_affect": "Cheerful, warm, bright voice with natural excitement",
        "tone": "Joyful, friendly, and expressive tone with emotional warmth",
        "pacing": "Lively energetic pacing with smooth transitions",
        "emotion_description": "Happiness, excitement, friendliness, comfort",
        "personality": "Joyful, open-hearted, lively",
        "pauses": "Short expressive pauses that feel playful"
    },
    "sad": {
        "voice_affect": "Soft, emotional voice with gentle heaviness",
        "tone": "Low, muted tone carrying emotional sensitivity",
        "pacing": "Slow pacing with long emotional pauses",
        "emotion_description": "Sorrow, grief, emotional pain, or disappointment",
        "personality": "Gentle, sensitive, introspective",
        "pauses": "Deep pauses that reflect sadness and emotional weight"
    },
    "anger": {
        "voice_affect": "Strong, sharp emotional color with harsh intensity",
        "tone": "Firm, forceful, irritated tone with rising tension",
        "pacing": "Fast, pressured delivery with strong emphasis",
        "emotion_description": "Anger, frustration, conflict, or irritation",
        "personality": "Direct, assertive, bold",
        "pauses": "Short harsh pauses that emphasize aggression"
    },
    "fear": {
        "voice_affect": "Shaky, nervous emotional color with audible tension",
        "tone": "Anxious, hesitant tone with unstable pitch",
        "pacing": "Uneven pacing with sudden brief pauses",
        "emotion_description": "Fear, nervousness, uncertainty",
        "personality": "Timid, cautious, worried",
        "pauses": "Fragmented pauses showing fear"
    },
    "surprise": {
        "voice_affect": "Excited, alert voice with sudden emotional rise",
        "tone": "High-pitched and expressive tone",
        "pacing": "Fast bursts of speech showing excitement",
        "emotion_description": "Shock, amazement, unexpected reactions",
        "personality": "Expressive, reactive, curious",
        "pauses": "Quick dramatic pauses that enhance surprise"
    },
    "neutral": {
        "voice_affect": "Calm, balanced voice without emotional coloring",
        "tone": "Even, professional, and natural tone",
        "pacing": "Steady pacing with smooth delivery",
        "emotion_description": "Neutral, objective, and emotionally stable",
        "personality": "Professional, steady, composed",
        "pauses": "Minimal regular pauses"
    }
}

# ================================================================
# 3️⃣ EMOTION PREDICT + TTS FUNCTIONS
# ================================================================
# PREDICT EMOTION + FETCH META
# ================================================================
def predict_emotion(text, lang):
    cleaned = clean_text(text, lang)

    if lang == "si":
        tokenizer = si_tokenizer
        model = si_model
        max_len = SINHALA_MAX_LEN
        label_map = si_id2label
        meta_map = SI_EMOTION_META
    else:
        tokenizer = ta_tokenizer
        model = ta_model
        max_len = TAMIL_MAX_LEN
        label_map = ta_id2label
        meta_map = TA_EMOTION_META

    # Tokenize input
    enc = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding="max_length", max_length=max_len
    )

    # Predict logits
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).logits

    pred_id = torch.argmax(logits, dim=1).item()

    # Get predicted emotion label
    raw_emotion = label_map[str(pred_id)] if lang == "si" else label_map[pred_id]
    raw_emotion = raw_emotion.strip().lower()

    # Normalize to match meta keys exactly
    emotion_key_map = {
        "anger": "anger",
        "ang": "anger",
        "fear": "fear",
        "sadness": "sad",
        "sad":"sad",
        "happ": "happy",
        "happy": "happy",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    emotion = emotion_key_map.get(raw_emotion, "neutral")

    # Fetch meta for the emotion
    meta = meta_map[emotion]

    # Include emotion in meta for TTS instructions
    meta["emotion_name"] = emotion  

    return {"text": text, "emotion": emotion, **meta}


# ================================================================
# GENERATE TTS BASED ON META
# ================================================================
def generate_tts(text, meta, lang):
    # Use emotion_name key to avoid collision with emotion_description
    instructions = (
        f"Affect: {meta['voice_affect']}\n"
        f"Tone: {meta['tone']}\n"
        f"Pacing: {meta['pacing']}\n"
        f"Personality: {meta['personality']}\n"
        f"Pauses: {meta['pauses']}\n"
        f"Emotion: {meta['emotion_description']}"
    )

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions=instructions
    )

    # Save audio
    filename = f"{uuid4().hex}.wav"
    folder = SINHALA_TTS_DIR if lang == "si" else TAMIL_TTS_DIR
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(response.read())

    return filename


# ================================================================
# FASTAPI
# ================================================================
router = APIRouter()


###############################################################
# SINHALA ROUTES
###############################################################
@router.post("/sinhala/predict-emotion")
async def sinhala_predict(text: str = Form(...)):
    result = predict_emotion(text, lang="si")
    return JSONResponse(result)

@router.post("/sinhala/predict-emotion-tts")
async def sinhala_predict_tts(text: str = Form(...)):
    result = predict_emotion(text, lang="si")
    audio = generate_tts(result["text"], result, "si")
    return JSONResponse({
        "success": True,
        "classification": result,
        "audio_url": f"http://localhost:8000/audio/sinhala/{audio}"
    })

@router.get("/audio/sinhala/{filename}")
async def sinhala_audio(filename: str):
    file = os.path.join(SINHALA_TTS_DIR, filename)
    return FileResponse(file, media_type="audio/wav")


###############################################################
# TAMIL ROUTES
###############################################################
@router.post("/tamil/predict-emotion")
async def tamil_predict(text: str = Form(...)):
    result = predict_emotion(text, lang="ta")
    return JSONResponse(result)

@router.post("/tamil/predict-emotion-tts")
async def tamil_predict_tts(text: str = Form(...)):
    result = predict_emotion(text, lang="ta")
    audio = generate_tts(result["text"], result, "ta")
    return JSONResponse({
        "success": True,
        "classification": result,
        "audio_url": f"http://localhost:8000/audio/tamil/{audio}"
    })

@router.get("/audio/tamil/{filename}")
async def tamil_audio(filename: str):
    file = os.path.join(TAMIL_TTS_DIR, filename)
    return FileResponse(file, media_type="audio/wav")
