"""
VoiceNotes — Speech-to-text note-taking app
Backend: Flask + sounddevice + faster-whisper

Run:   python app.py
Open:  http://localhost:5000
"""

import sys
import os
import datetime
import warnings

# ── Suppress HuggingFace warnings ──────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, jsonify, send_file

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS    = 1
NOTES_DIR   = "notes"
MODEL_SIZE  = "base"

app = Flask(__name__)

# ── Recording state ────────────────────────────────────────────────────────────
_rec = {"active": False, "chunks": [], "stream": None}

# ── Lazy-loaded Whisper model (loads on first transcription) ───────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        print("  * Loading Whisper model (first time, ~150 MB download)...")
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("  * Model ready")
    return _model


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    if _rec["active"]:
        return jsonify({"error": "Already recording"}), 400

    _rec["chunks"] = []
    _rec["active"] = True

    def _audio_callback(indata, frames, time_info, status):
        if _rec["active"]:
            _rec["chunks"].append(indata.copy())

    try:
        _rec["stream"] = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=_audio_callback,
        )
        _rec["stream"].start()
    except Exception as exc:
        _rec["active"] = False
        return jsonify({"error": f"Microphone error: {exc}"}), 500

    return jsonify({"status": "recording"})


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    if not _rec["active"]:
        return jsonify({"error": "Not recording"}), 400

    _rec["active"] = False
    if _rec["stream"]:
        _rec["stream"].stop()
        _rec["stream"].close()
        _rec["stream"] = None

    if not _rec["chunks"]:
        return jsonify({"error": "No audio captured"}), 400

    # Save temp WAV
    audio = np.concatenate(_rec["chunks"], axis=0)
    tmp = "_tmp_rec.wav"
    sf.write(tmp, audio, SAMPLE_RATE)

    # Transcribe directly (no multiprocessing needed!)
    try:
        model = _get_model()
        segs, info = model.transcribe(tmp)
        text = " ".join(s.text.strip() for s in segs).strip()
        lang = info.language
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if not text:
        return jsonify({"error": "No speech detected"}), 400

    # ── Save note ──────────────────────────────────────────────────────────────
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    day_dir  = os.path.join(NOTES_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)
    fname = f"note_{time_str}.txt"
    fpath = os.path.join(day_dir, fname)

    lang_map   = {"ar": "Arabic", "en": "English"}
    lang_label = lang_map.get(lang, lang.upper())
    header = (
        f"Date : {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Lang : {lang_label}\n"
        f"{'─' * 40}\n\n"
    )
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(header + text)

    return jsonify({
        "text": text,
        "language": lang_label,
        "filename": fname,
        "date": date_str,
    })


@app.route("/api/notes")
def list_notes():
    result = []
    if os.path.isdir(NOTES_DIR):
        for date in sorted(os.listdir(NOTES_DIR), reverse=True):
            dp = os.path.join(NOTES_DIR, date)
            if not os.path.isdir(dp):
                continue
            files = sorted(
                [f for f in os.listdir(dp) if f.endswith(".txt")],
                reverse=True,
            )
            if files:
                result.append({"date": date, "files": files})
    return jsonify(result)


@app.route("/api/notes/<date>/<filename>")
def get_note(date, filename):
    fpath = os.path.join(NOTES_DIR, date, filename)
    if not os.path.isfile(fpath):
        return jsonify({"error": "Not found"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"content": content, "filename": filename, "date": date})


@app.route("/api/notes/<date>/<filename>", methods=["DELETE"])
def delete_note(date, filename):
    fpath = os.path.join(NOTES_DIR, date, filename)
    if not os.path.isfile(fpath):
        return jsonify({"error": "Not found"}), 404
    os.remove(fpath)
    dp = os.path.join(NOTES_DIR, date)
    if not os.listdir(dp):
        os.rmdir(dp)
    return jsonify({"status": "deleted"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(NOTES_DIR, exist_ok=True)
    print()
    print("  ========================================")
    print("           VoiceNotes is running          ")
    print("       http://localhost:5000              ")
    print("  ========================================")
    print()

    import webbrowser
    webbrowser.open("http://localhost:5000")

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
