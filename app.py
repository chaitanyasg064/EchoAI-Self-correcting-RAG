import logging
import os
import threading
import uuid
from functools import wraps
from typing import Optional

import pyttsx3
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file, session
from flask_cors import CORS
from groq import Groq

import db
from coversational_rag import ConversationalRAG

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("echoai.app")

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

db.init_db()

file_directory = "documentation/"
file_paths = []
if os.path.isdir(file_directory):
    file_paths = [
        os.path.join(file_directory, f)
        for f in os.listdir(file_directory)
        if f.endswith(".csv") or f.endswith(".pdf")
    ]
else:
    logger.warning("Documentation directory '%s' not found; knowledge base will be empty.", file_directory)

conversational_rag = ConversationalRAG(file_paths=file_paths, api_key=os.environ.get("GROQ_API_KEY"))

# --- Per-conversation request locking (never process two requests for the same chat at once) ---
_locks_guard = threading.Lock()
_busy_conversations: set = set()


def _try_acquire(key: str) -> bool:
    with _locks_guard:
        if key in _busy_conversations:
            return False
        _busy_conversations.add(key)
        return True


def _release(key: str) -> None:
    with _locks_guard:
        _busy_conversations.discard(key)


def handle_errors(default_message: str):
    """Outer safety net: logs the full traceback and returns a clean JSON error instead of crashing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception("Unhandled error in %s", func.__name__)
                return jsonify({"error": default_message}), 500
        return wrapper
    return decorator


def _resolve_conversation_id(conversation_id: Optional[str]) -> str:
    if conversation_id and db.get_conversation(conversation_id):
        return conversation_id
    conv = db.create_conversation(title="New Chat")
    return conv["id"]


def _run_rag_turn(text: str, conversation_id: str) -> str:
    """Hydrates memory if needed, runs the RAG pipeline, and persists both turns."""
    existing_messages = db.get_messages(conversation_id)
    conversational_rag.hydrate_session(conversation_id, [(m["role"], m["content"]) for m in existing_messages])

    answer = conversational_rag.qa_with_memory(text, conversation_id)[0]

    db.add_message(conversation_id, "user", text)
    db.add_message(conversation_id, "assistant", answer)

    if not existing_messages:
        db.rename_conversation(conversation_id, db.generate_title_from_text(text))

    return answer


@app.before_request
def initialize_user_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())


# ==========================================================================================
# Conversation management (ChatGPT-style sidebar)
# ==========================================================================================
@app.route("/api/conversations", methods=["GET"])
@handle_errors("Could not load conversations.")
def api_list_conversations():
    return jsonify({"conversations": db.list_conversations()}), 200


@app.route("/api/conversations", methods=["POST"])
@handle_errors("Could not create a new conversation.")
def api_create_conversation():
    data = request.get_json(silent=True) or {}
    title = data.get("title") or "New Chat"
    conv = db.create_conversation(title=title)
    return jsonify(conv), 201


@app.route("/api/conversations/<conversation_id>/messages", methods=["GET"])
@handle_errors("Could not load conversation messages.")
def api_get_messages(conversation_id):
    if not db.get_conversation(conversation_id):
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"messages": db.get_messages(conversation_id)}), 200


@app.route("/api/conversations/<conversation_id>", methods=["PATCH"])
@handle_errors("Could not rename conversation.")
def api_rename_conversation(conversation_id):
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title cannot be empty."}), 400
    if not db.rename_conversation(conversation_id, title):
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"id": conversation_id, "title": title}), 200


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
@handle_errors("Could not delete conversation.")
def api_delete_conversation(conversation_id):
    if not db.delete_conversation(conversation_id):
        return jsonify({"error": "Conversation not found."}), 404
    conversational_rag.forget_session(conversation_id)
    return jsonify({"deleted": True}), 200


@app.route("/api/conversations/search", methods=["GET"])
@handle_errors("Search failed.")
def api_search_conversations():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"conversations": db.list_conversations()}), 200
    return jsonify({"conversations": db.search_conversations(query)}), 200


# ==========================================================================================
# Core assistant endpoints
# ==========================================================================================
@app.route("/ask", methods=["POST"])
@handle_errors("An internal error occurred while answering the question.")
def ask():
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    user_session_id = session["session_id"]
    answer = conversational_rag.qa_with_memory(user_question, user_session_id)
    return {"answer": answer}


@app.route("/llm", methods=["POST"])
@handle_errors("An internal error occurred while generating a response.")
def qa_with_memory():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    conversation_id = _resolve_conversation_id(data.get("conversation_id"))

    if not _try_acquire(conversation_id):
        return jsonify({"error": "A request is already in progress for this conversation.", "conversation_id": conversation_id}), 409

    try:
        answer = _run_rag_turn(text, conversation_id)
        return jsonify({"text": answer, "conversation_id": conversation_id}), 200
    finally:
        _release(conversation_id)


@app.route("/ai", methods=["POST"])
@handle_errors("An internal error occurred.")
def generate_llm_response():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    conversation_id = _resolve_conversation_id(data.get("conversation_id"))

    if not _try_acquire(conversation_id):
        return jsonify({"error": "A request is already in progress for this conversation.", "conversation_id": conversation_id}), 409

    try:
        answer = _run_rag_turn(text, conversation_id)
        return jsonify({"text": answer, "conversation_id": conversation_id}), 200
    finally:
        _release(conversation_id)


def synthesize_speech(text: str, output_path: str = "output.wav") -> None:
    """Offline TTS via pyttsx3 — no external API, no character limits, no API key."""
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()
    except Exception:
        logger.exception("pyttsx3 TTS generation failed for output_path=%s", output_path)
        raise


@app.route("/stt", methods=["POST"])
@handle_errors("Speech-to-text failed.")
def stt():
    filename = "temp_audio.wav"
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        audio_file.save(filename)

        with open(filename, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                prompt="Specify context or spelling",
                response_format="json",
                language="en",
                temperature=0.0,
            )
        return jsonify({"text": transcription.text})
    finally:
        if os.path.exists(filename):
            os.remove(filename)


@app.route("/sts", methods=["POST"])
@handle_errors("Speech-to-speech failed.")
def sts():
    filename = "temp_audio.wav"
    audio_path = "output.wav"
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        audio_file.save(filename)

        with open(filename, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                prompt="Specify context or spelling",
                response_format="json",
                language="en",
                temperature=0.0,
            )

        conversation_id = _resolve_conversation_id(request.form.get("conversation_id"))

        if not _try_acquire(conversation_id):
            return jsonify({"error": "A request is already in progress for this conversation.", "conversation_id": conversation_id}), 409

        try:
            llm_response = _run_rag_turn(transcription.text, conversation_id)

            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except PermissionError:
                    pass

            try:
                synthesize_speech(llm_response, audio_path)
            except Exception as e:
                return jsonify({"error": f"Text-to-speech generation failed: {str(e)}", "conversation_id": conversation_id}), 502

            response = send_file(audio_path, mimetype="audio/wav", as_attachment=False)
            response.headers["X-Conversation-Id"] = conversation_id
            response.headers["X-Transcript"] = transcription.text
            response.headers["X-Answer"] = llm_response
            return response
        finally:
            _release(conversation_id)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


@app.route("/tts", methods=["POST"])
@handle_errors("Text-to-speech failed.")
def tts():
    audio_path = "output.wav"
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except PermissionError:
            pass

    try:
        synthesize_speech(text, audio_path)
    except Exception as e:
        return jsonify({"error": f"Text-to-speech generation failed: {str(e)}"}), 502

    return send_file(audio_path, mimetype="audio/wav", as_attachment=False)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)