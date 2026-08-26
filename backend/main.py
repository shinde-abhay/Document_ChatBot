import os
import psycopg2
from psycopg2 import errors as pg_errors
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load .env file from the parent directory
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


# --- RAG Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "..", "faiss_index")
UPLOAD_DIR = os.path.join(BASE_DIR, "user_uploads")
USER_INDEX_DIR = os.path.join(BASE_DIR, "user_indexes")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(USER_INDEX_DIR, exist_ok=True)
user_chains = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://document-chatbot-frontend-prod.up.railway.app",  # Railway prod
        "http://localhost:8501",                                   # Local dev
        "http://localhost:3000",                                   # Alternative local
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)

SYSTEM_PROMPT = '''
You are a helpful assistant for a document Q&A chatbot.
Use the conversation history below to remember what the user has told you (like their name) and to understand follow-up questions.
Use the retrieved context to answer questions about the documents (Java OOP, web scraping).
Answer in max three sentences. If the answer isn't in the context or conversation history, say you don't know.

Context: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
def get_db_conn():
    conn = psycopg2.connect(DB_URL)
    return conn


class QueryRequest(BaseModel):
    user_id: int
    text: str


class HistoryRequest(BaseModel):
    user_id: int


class UserRequest(BaseModel):
    username: str

#api endpoint

#login/signup
# NOTE: Demo project — this uses simple username-based sessions with no
# password authentication. Not suitable for handling real user data.
# A production version would add password auth (hashing + login) or OAuth.

@app.post("/get_or_create_user")
def get_or_create_user(req: UserRequest):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # 1. Try to find the user
        cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
        user_row = cur.fetchone()

        if user_row:
            user_id = user_row[0]
        else:
            try:
                # 2. If not found, try to create them
                cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING id", (req.username,))
                conn.commit()
                user_id = cur.fetchone()[0]
            except pg_errors.UniqueViolation:
                # Someone else created this username in the meantime — fetch it instead
                conn.rollback()
                cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
                user_id = cur.fetchone()[0]

        return {"user_id": user_id, "username": req.username}
    finally:
        cur.close()
        conn.close()

#chat history

@app.post("/get_history")
def get_history(req: HistoryRequest):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC", (req.user_id,))
        history = cur.fetchall()

        formatted_history = []
        for p, a in history:
            formatted_history.append({"role": "human", "content": p})
            formatted_history.append({"role": "ai", "content": a})

        return {"history": formatted_history}
    finally:
        cur.close()
        conn.close()

def get_loader_for_file(file_path: str, filename: str):
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(file_path)
    elif ext == "docx":
        return Docx2txtLoader(file_path)
    elif ext == "txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == "csv":
        return CSVLoader(file_path)
    else:
        return None

def get_user_chain(user_id: int):
    # 1. Already cached in memory? Use it.
    if user_id in user_chains:
        return user_chains[user_id]

    # 2. Not cached (e.g. server restarted) — try loading their saved index from disk.
    user_index_path = os.path.join(USER_INDEX_DIR, str(user_id))
    if os.path.exists(user_index_path):
        user_db = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = user_db.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        user_chains[user_id] = rag_chain
        return rag_chain

    # 3. No index at all — they haven't uploaded a document yet.
    return None

@app.post("/upload")
async def upload_file(user_id: int = Form(...), file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.lower().split(".")[-1]

    if ext not in ("pdf", "docx", "txt", "csv"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, TXT, or CSV.")

    # Save the uploaded file to disk
    user_upload_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Load and parse the file based on its type
    loader = get_loader_for_file(file_path, filename)
    try:
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not docs:
        raise HTTPException(status_code=400, detail="No readable text found in the file.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    # Build a fresh FAISS index for this user and save it to disk
    user_db = FAISS.from_documents(chunks, embeddings)
    user_index_path = os.path.join(USER_INDEX_DIR, str(user_id))
    user_db.save_local(user_index_path)

    # Build this user's retrieval chain and cache it in memory
    retriever = user_db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    user_chains[user_id] = rag_chain

    return {"message": f"'{filename}' uploaded and indexed successfully.", "chunks_indexed": len(chunks)}

@app.post("/query")
def query_rag(req: QueryRequest):
    rag_chain = get_user_chain(req.user_id)
    if rag_chain is None:
        return {"answer": "You haven't uploaded a document yet. Please upload a PDF, DOCX, TXT, or CSV file first."}

    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC", (req.user_id,))
        db_history = cur.fetchall()

        chat_history_messages = []
        for prompt, answer in db_history:
            chat_history_messages.append(HumanMessage(content=prompt))
            chat_history_messages.append(AIMessage(content=answer))

        response = rag_chain.invoke({
            "input": req.text,
            "chat_history": chat_history_messages
        })
        answer = response.get("answer", "No answer found.")

        # Save new Q&A to database
        cur.execute("INSERT INTO chat_history (user_id, prompt, answer) VALUES (%s, %s, %s)", (req.user_id, req.text, answer))
        conn.commit()

        return {"answer": answer}
    finally:
        cur.close()
        conn.close()

@app.get("/")
def read_root():
    return {"message": "welcome to fastapi.go to /docs to get started"}
