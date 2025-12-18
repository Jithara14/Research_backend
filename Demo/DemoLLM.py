import time, sys, collections
import requests, webrtcvad, sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import simpleaudio as sa
import io
from llama_cpp import Llama
import json

ELEVEN_API_KEY = "sk_9437576b5db9da5ec979cb20e0ea435071261c51e8a16a4a"
VOICE_ID = "6qq2wIqsoWwSH2ed23B2"
TTS_MODEL = "eleven_multilingual_v2"

STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_MODEL = "scribe_v1"

MODEL_PATH = "models/tamil-llama-7b-q8_0.gguf"

print("Loading LLaMA model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=20,
    verbose=False
)
print("LLaMA ready.\n")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

FRAME_MS = 10
BYTES_PER_S = 2
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * BYTES_PER_S
BLOCKSIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_AGGR = 3
CALIBRATION_SEC = 1.3
START_RMS_FACTOR = 4.0
STOP_RMS_FACTOR = 1.6
TRIGGER_VOICED_FRAMES = 10
QUIET_STOP_FRAMES = 80
MAX_UTT_SEC = 20
OUTPUT_WAV = "input.wav"


def rms_int16(pcm: bytes) -> float:
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float32)**2)))


def calibrate_noise():
    print(f"Calibrating noise for {CALIBRATION_SEC}s...")
    end = time.time() + CALIBRATION_SEC
    rms_vals = []

    def cb(indata, frames, t, status):
        buf = bytes(indata)
        for i in range(0, len(buf), FRAME_BYTES):
            frame = buf[i:i+FRAME_BYTES]
            if len(frame) == FRAME_BYTES:
                rms_vals.append(rms_int16(frame))

    with sd.RawInputStream(samplerate=SAMPLE_RATE,
                           blocksize=BLOCKSIZE,
                           channels=CHANNELS,
                           dtype=DTYPE,
                           callback=cb):
        while time.time() < end:
            sd.sleep(50)

    baseline = float(np.median(rms_vals)) if rms_vals else 80.0
    print(f"Noise Level: {baseline:.1f}")
    return baseline


def record_until_auto_stop():
    vad = webrtcvad.Vad(VAD_AGGR)

    baseline = calibrate_noise()
    start_thr = max(120.0, baseline * START_RMS_FACTOR)
    stop_thr = max(80.0, baseline * STOP_RMS_FACTOR)

    print(f"Start if RMS > {start_thr:.1f} | Stop if RMS < {stop_thr:.1f}")
    print("Listening... Speak anytime.")

    buf = bytearray()
    recording = False
    voice_frames = 0
    quiet_frames = 0
    captured = []
    total_ms = 0

    preroll = collections.deque(maxlen=20)

    def callback(indata, frames, t, status):
        nonlocal buf, recording, voice_frames, quiet_frames, captured, total_ms

        buf.extend(bytes(indata))

        while len(buf) >= FRAME_BYTES:
            frame = bytes(buf[:FRAME_BYTES])
            del buf[:FRAME_BYTES]
            total_ms += FRAME_MS

            energy = rms_int16(frame)
            vad_flag = vad.is_speech(frame, SAMPLE_RATE)

            if not recording:
                preroll.append(frame)

                if vad_flag and energy > start_thr:
                    voice_frames += 1
                    if voice_frames >= TRIGGER_VOICED_FRAMES:
                        recording = True
                        captured.extend(list(preroll))
                        preroll.clear()
                        print("Recording started.")
                else:
                    voice_frames = 0

            else:
                captured.append(frame)

                if energy < stop_thr:
                    quiet_frames += 1
                    if quiet_frames >= QUIET_STOP_FRAMES:
                        print("Silent detected ! Stopping.")
                        raise sd.CallbackStop()
                else:
                    quiet_frames = 0

            if total_ms >= MAX_UTT_SEC * 1000:
                print("Max duration reached.")
                raise sd.CallbackStop()

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=BLOCKSIZE,
                               channels=CHANNELS,
                               dtype=DTYPE,
                               callback=callback):
            sd.sleep(MAX_UTT_SEC * 1000 + 2000)
    except sd.CallbackStop:
        pass

    if not captured:
        return b""

    return b"".join(captured)


def save_wav(pcm_bytes: bytes, path: str):
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    write(path, SAMPLE_RATE, arr)


def stt_elevenlabs(wav_path: str) -> str:
    headers = {"xi-api-key": ELEVEN_API_KEY}
    with open(wav_path, "rb") as f:
        files = {"file": (wav_path, f, "audio/wav")}
        data = {"model_id": STT_MODEL}
        res = requests.post(STT_URL, headers=headers, data=data, files=files)

    if res.status_code != 200:
        print("STT Error:", res.text)
        return ""

    return res.json().get("text", "")


def llm_reply(prompt: str) -> str:
    response = llm(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.95
    )
    return response["choices"][0]["text"]


def tts_elevenlabs(text: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "Accept": "audio/wav",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY,
    }

    payload = {
        "text": text,
        "model_id": TTS_MODEL,
        "output_format": "wav_44100_16"
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

    if r.status_code != 200:
        print("TTS error:", r.text)
        return

    ctype = r.headers.get("Content-Type", "")
    audio_bytes = r.content

    if audio_bytes.startswith(b"RIFF"):
        with open("reply.wav", "wb") as f:
            f.write(audio_bytes)
    else:
        mp3_buf = io.BytesIO(audio_bytes)
        seg = AudioSegment.from_file(mp3_buf, format="mp3")
        wav_buf = io.BytesIO()
        seg.set_frame_rate(44100).set_sample_width(2).export(wav_buf, format="wav")
        with open("reply.wav", "wb") as f:
            f.write(wav_buf.getvalue())

    wave_obj = sa.WaveObject.from_wave_file("reply.wav")
    play = wave_obj.play()
    play.wait_done()


print("Tamil Voice LLM Assistant Ready\n")

while True:
    print("Speak now... (Ctrl+C to exit)\n")

    pcm = record_until_auto_stop()
    if not pcm:
        print("No speech detected.\n")
        continue

    save_wav(pcm, OUTPUT_WAV)

    print("Converting speech to text...")
    text = stt_elevenlabs(OUTPUT_WAV)
    print(f"You said: {text}\n")

    print("LLM thinking...")
    reply = llm_reply(text)
    print(f"LLM Reply: {reply}\n")

    print("Speaking reply...\n")
    tts_elevenlabs(reply)
