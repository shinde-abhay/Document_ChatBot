# DocChat — Personalized Document Chatbot

A full-stack RAG (Retrieval-Augmented Generation) application that lets users upload their own documents — PDF, DOCX, TXT, or CSV — and chat with a personalized AI assistant built on top of that content. Each user gets their own isolated knowledge base, persistent chat history, and conversational memory.

## Features

- **Bring your own document** — upload a PDF, Word doc, text file, or CSV and get an instant, personalized chatbot for it
- **Per-user isolation** — every user's uploaded document is embedded into a separate FAISS vector index, so knowledge bases never mix between users
- **Conversational memory** — the chatbot remembers context within a conversation (e.g. your name, earlier questions) using proper chat-history-aware prompting
- **Persistent chat history** — all conversations are saved to PostgreSQL and reloaded on login
- **Simple session system** — lightweight username-based sessions (see [Known Limitations](#known-limitations))

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Vector store | FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| Orchestration | LangChain |
| Database | PostgreSQL (`psycopg2`) |
| Document parsing | `pypdf`, `docx2txt`, LangChain loaders |

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────┐
│  Streamlit  │ ───────────────▶│   FastAPI     │
│  Frontend   │◀─────────────── │   Backend     │
└─────────────┘                 └───────┬──────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
             │ PostgreSQL  │      │ FAISS Index   │      │ Gemini API   │
             │ (users +    │      │ (per-user,    │      │ (LLM         │
             │  history)   │      │  on disk)     │      │  responses)  │
             └─────────────┘      └──────────────┘      └──────────────┘
```

**Flow:**
1. User logs in (username-based session) → session and chat history loaded from PostgreSQL
2. User uploads a document → text extracted, chunked, embedded, and saved as a FAISS index unique to that user
3. User asks a question → relevant chunks retrieved from their FAISS index, combined with chat history, sent to Gemini → answer returned and saved to PostgreSQL

## Project Structure

```
ChatBot/
├── backend/
│   ├── main.py          # FastAPI app: endpoints, RAG chain, upload logic
│   ├── tables.py         # One-time DB table setup
│   ├── user_uploads/      # Uploaded files, per user (gitignored)
│   └── user_indexes/      # FAISS indexes, per user (gitignored)
├── frontend/
│   └── app.py             # Streamlit chat UI
├── .env                    # API keys & DB connection string (gitignored)
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd ChatBot
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Set up PostgreSQL

Create a database:
```sql
CREATE DATABASE chatbot_db;
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot_db
```

### 5. Create database tables

```bash
python backend/tables.py
```

You should see `Database connection successful` and `Tables created successfully`.

### 6. Run the backend

```bash
cd backend
uvicorn main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

### 7. Run the frontend

In a separate terminal:

```bash
cd frontend
streamlit run app.py
```

## Usage

1. Log in with any username (creates an account automatically on first use)
2. Upload a PDF, DOCX, TXT, or CSV file from the sidebar
3. Start asking questions — the assistant answers using only the content of your uploaded document
4. Uploading a new file replaces your previous document and starts a fresh chat

## Known Limitations

- **No password authentication.** Sessions are username-based only, with no password check. This is a deliberate scope decision for a demo/portfolio project — a production version would add password hashing, JWT sessions, or OAuth.
- **CSV support is text-oriented.** RAG-based retrieval works well for text-heavy CSVs (e.g. FAQs, notes) but isn't suited to numeric/aggregate questions ("what's the total revenue?") — that would require a SQL or pandas-based approach instead of vector search.
- **One active document per user.** Uploading a new file replaces the previous one rather than merging into a combined knowledge base. Multi-document support is a natural next step.

## Possible Future Improvements

- Multi-document knowledge bases per user
- Password-based authentication
- Support for additional file types (PPTX, XLSX, Markdown)
- Streaming responses for faster perceived latency
- Deployment guide (Docker, cloud hosting)

## License

This project is open for personal and educational use.
