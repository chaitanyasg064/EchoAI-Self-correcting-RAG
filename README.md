# EchoAI

> **Enterprise Voice Assistant powered by Self-Correcting RAG, Hybrid Retrieval, Qdrant, and Conversation Memory.**

EchoAI is a voice-first AI assistant designed for enterprise knowledge retrieval. It answers questions using organization-specific documents through a Self-Correcting Retrieval-Augmented Generation (RAG) pipeline and seamlessly falls back to a general LLM when no relevant knowledge is available.

---

## Features

- 🎤 Voice-to-Text using Groq Whisper
- 🧠 Self-Correcting RAG Pipeline
- 🔎 Hybrid Retrieval (Qdrant + General LLM)
- 💬 Persistent Conversation Memory
- 📚 Multi-document Support (CSV & PDF)
- 🗂 ChatGPT-style Chat History
- 🔊 Offline Text-to-Speech
- ⚡ Low-latency Responses
- 🛡 Robust Error Handling
- 🔒 Session-based Conversations

---

## Architecture

```
          Voice / Text Input
                  │
          Whisper Speech-to-Text
                  │
          Hybrid Retrieval Router
        ┌─────────┴─────────┐
        │                   │
     Qdrant RAG        General LLM
        │                   │
        └─────────┬─────────┘
                  │
      Conversation Memory
                  │
         Text / Speech Output
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| LLM | Groq Llama |
| STT | Groq Whisper |
| TTS | pyttsx3 |
| Vector Database | Qdrant |
| Embeddings | Sentence Transformers |
| Memory | SQLite |
| Framework | LangChain |

---

## Project Structure

```
EchoAI/
│── app.py
│── conversational_rag.py
│── db.py
│── requirements.txt
│── documentation/
│── templates/
│── static/
│── qdrant_storage/
│── Dockerfile
│── README.md
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd EchoAI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_api_key
```

---

## Adding Documents

Place your PDF or CSV files inside:

```
documentation/
```

The assistant automatically indexes them into Qdrant on startup.

---

## Run

```bash
python app.py
```

Open:

```
http://localhost:5000
```

---

## Docker

Build

```bash
docker build -t echoai .
```

Run

```bash
docker run -p 5000:5000 --env-file .env echoai
```

---

## Future Improvements

- Authentication
- Streaming Responses
- Multi-language Support
- Cloud Deployment
- Admin Dashboard
- Additional Knowledge Sources

---

## Author
Chaitanya S G