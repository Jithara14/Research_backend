import time, collections, io, json
import requests, webrtcvad, sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import simpleaudio as sa

# -------------------------
# CONFIG (unchanged)
# -------------------------
ELEVEN_API_KEY = "sk_9437576b5db9da5ec979cb20e0ea435071261c51e8a16a4a"
VOICE_ID = "6qq2wIqsoWwSH2ed23B2"
TTS_MODEL = "eleven_multilingual_v2"

STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_MODEL = "scribe_v1"

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


# -------------------------
# Utility functions
# -------------------------
def rms_int16(pcm: bytes) -> float:
    arr = np.frombuffer(pcm, dtype=np.int16)
    return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if arr.size else 0.0


def calibrate_noise():
    end = time.time() + CALIBRATION_SEC
    rms_vals = []

    def cb(indata, frames, t, status):
        buf = bytes(indata)
        for i in range(0, len(buf), FRAME_BYTES):
            frame = buf[i:i+FRAME_BYTES]
            if len(frame) == FRAME_BYTES:
                rms_vals.append(rms_int16(frame))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=cb
    ):
        while time.time() < end:
            sd.sleep(50)

    return float(np.median(rms_vals)) if rms_vals else 80.0


# -------------------------
# Record until silence
# -------------------------
def record_until_auto_stop():
    vad = webrtcvad.Vad(VAD_AGGR)
    baseline = calibrate_noise()
    start_thr = max(120.0, baseline * START_RMS_FACTOR)
    stop_thr = max(80.0, baseline * STOP_RMS_FACTOR)

    buf = bytearray()
    recording = False
    voice_frames = quiet_frames = total_ms = 0
    captured = []
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
                else:
                    voice_frames = 0
            else:
                captured.append(frame)
                if energy < stop_thr:
                    quiet_frames += 1
                    if quiet_frames >= QUIET_STOP_FRAMES:
                        raise sd.CallbackStop()
                else:
                    quiet_frames = 0

            if total_ms >= MAX_UTT_SEC * 1000:
                raise sd.CallbackStop()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback
        ):
            sd.sleep(MAX_UTT_SEC * 1000 + 2000)
    except sd.CallbackStop:
        pass

    return b"".join(captured)


def save_wav(pcm_bytes):
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    write(OUTPUT_WAV, SAMPLE_RATE, arr)


# -------------------------
# Speech → Text (ElevenLabs)
# -------------------------
def speech_to_text():
    pcm = record_until_auto_stop()
    if not pcm:
        return ""

    save_wav(pcm)

    headers = {"xi-api-key": ELEVEN_API_KEY}
    with open(OUTPUT_WAV, "rb") as f:
        res = requests.post(
            STT_URL,
            headers=headers,
            data={"model_id": STT_MODEL},
            files={"file": f}
        )

    if res.status_code != 200:
        return ""

    return res.json().get("text", "")


# -------------------------
# Text → Speech (ElevenLabs)
# -------------------------
def speak(text: str):
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

    audio_bytes = r.content

    # ✅ CASE 1: Proper WAV (RIFF)
    if audio_bytes[:4] == b"RIFF":
        with open("reply.wav", "wb") as f:
            f.write(audio_bytes)

    # ✅ CASE 2: MP3 → convert to WAV
    else:
        mp3_buf = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(mp3_buf, format="mp3")
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
        audio.export("reply.wav", format="wav")

    wave_obj = sa.WaveObject.from_wave_file("reply.wav")
    play = wave_obj.play()
    play.wait_done()
