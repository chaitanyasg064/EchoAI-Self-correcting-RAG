"""
conversational_rag.py

Self-Correcting Agentic RAG pipeline for EchoAI, with hybrid RAG/general-LLM
routing and summarizing conversation memory.

Pipeline:
    1. Contextualize the question against (summarized) chat history.
    2. Score retrieval relevance in Qdrant. If the top similarity score is
       below RAG_SIMILARITY_THRESHOLD, skip RAG entirely and answer with the
       base LLM in general-assistant mode (logged as mode=GENERAL_LLM).
    3. Otherwise, run the self-correcting RAG loop (evaluate → rewrite →
       retry, up to MAX_RETRIES) and answer only from retrieved context
       (logged as mode=RAG), falling back to a clarification question if
       retrieval never becomes sufficient.
    4. Conversation memory is periodically summarized so only a compact
       summary + the most recent turns are sent to the LLM, bounding token
       usage regardless of how long a conversation runs.
"""

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger("echoai.rag")

MAX_RETRIES = 2
RAG_SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.55"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MEMORY_MAX_MESSAGES", "12"))
SUMMARY_KEEP_RECENT = int(os.environ.get("MEMORY_KEEP_RECENT", "6"))
RETRIEVAL_CACHE_SIZE = 128

FINAL_ANSWER_SYSTEM_PROMPT = """You are EchoAI, a professional, friendly customer support assistant for Samsung products.

Answer the user's question using ONLY the information in the retrieved context below and the conversation history. \
Do not use outside knowledge and do not guess. If the context does not fully answer the question, say what you \
do know and ask a short follow-up question to narrow down the request.

Voice output rules (the response will be read aloud by text-to-speech):
- Keep the answer concise: at most 2 short sentences.
- Use plain spoken language. Do NOT use special characters, markdown, bullet points, numbers with symbols, \
or emojis, since these cannot be read aloud correctly.
- Be warm and professional.

Retrieved context:
{context}"""

GENERAL_SYSTEM_PROMPT = """You are EchoAI, a friendly, professional voice assistant. The user's question does not \
match anything in the product knowledge base, so answer using your own general knowledge.

Voice output rules (the response will be read aloud by text-to-speech):
- Keep the answer concise: at most 2 short sentences.
- Use plain spoken language. Do NOT use special characters, markdown, or emojis.
- Be warm and professional."""

CONTEXTUALIZE_SYSTEM_PROMPT = """Given the chat history and the latest user question, rewrite the question into a \
standalone question that can be understood without the chat history. Do NOT answer the question. If the question \
is already standalone, return it unchanged. Return ONLY the rewritten question, nothing else."""

EVALUATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict retrieval quality evaluator for a customer support system.

User question:
{question}

Retrieved context:
{context}

Decide whether the retrieved context contains enough specific information to answer the user's question \
accurately and completely, without guessing or adding outside knowledge.

Reply with exactly one word: ENOUGH or INSUFFICIENT."""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """The following search query did not retrieve enough relevant information from a Samsung product \
documentation knowledge base.

Original query:
{question}

Rewrite it as a single, more specific and semantically clear search query that is more likely to retrieve \
relevant passages. Expand abbreviations, add likely product/feature terminology, and remove conversational \
filler. Return ONLY the rewritten query, nothing else."""
)

CLARIFICATION_PROMPT = ChatPromptTemplate.from_template(
    """The knowledge base does not contain enough information to confidently answer the user's question below.

User question:
{question}

Ask exactly ONE short, specific clarification question that would help find the right answer. \
Keep it under 20 words, use plain spoken language with no special characters, since it will be read aloud."""
)

SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    """You maintain a running summary of a customer support conversation.

Previous summary:
{previous_summary}

New conversation turns to fold in:
{conversation}

Write an updated, concise summary (max 5 sentences) capturing all facts, preferences, and open questions \
needed to understand future messages. Do not lose concrete facts (product names, numbers, decisions). \
Return ONLY the updated summary."""
)


class SessionState:
    __slots__ = ("history", "summary")

    def __init__(self) -> None:
        self.history: ChatMessageHistory = ChatMessageHistory()
        self.summary: str = ""


class ConversationalRAG:
    def __init__(
        self,
        file_paths: List[str],
        api_key: str,
        model_name: str = "llama-3.1-8b-instant",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        logger.info("Initializing ConversationalRAG with %d source file(s)", len(file_paths))

        # fastembed uses ONNX runtime instead of torch — much lighter footprint,
        # and BAAI/bge-small-en-v1.5 is also 384-dim, matching the Qdrant collection.
        self.embed_model = FastEmbedEmbeddings(model_name=embedding_model)
        self.chat_model = ChatGroq(temperature=0, model_name=model_name, api_key=api_key)

        self.qdrant_client, self.collection_name = self._init_qdrant_client()
        self._ensure_collection_and_index(file_paths)

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embed_model,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

        self.contextualize_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            | self.chat_model
            | StrOutputParser()
        )

        self.evaluator_chain = EVALUATOR_PROMPT | self.chat_model | StrOutputParser()
        self.rewrite_chain = REWRITE_PROMPT | self.chat_model | StrOutputParser()
        self.clarification_chain = CLARIFICATION_PROMPT | self.chat_model | StrOutputParser()
        self.summarize_chain = SUMMARIZE_PROMPT | self.chat_model | StrOutputParser()

        self.answer_chain = create_stuff_documents_chain(
            self.chat_model,
            ChatPromptTemplate.from_messages(
                [
                    ("system", FINAL_ANSWER_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
        )

        self.general_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", GENERAL_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            | self.chat_model
            | StrOutputParser()
        )

        # Session-scoped memory: session_id -> SessionState (history + rolling summary).
        self.store: Dict[str, SessionState] = {}

        # Bounded cache of (docs, top_score) keyed by normalized question text,
        # to avoid re-embedding/re-searching identical or repeated queries.
        self._retrieval_cache: "OrderedDict[str, Tuple[List[Document], float]]" = OrderedDict()

    # ------------------------------------------------------------------------------------
    # Qdrant setup
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _init_qdrant_client() -> Tuple[QdrantClient, str]:
        collection_name = os.environ.get("QDRANT_COLLECTION", "echoai_documents")
        qdrant_url = os.environ.get("QDRANT_URL")
        try:
            if qdrant_url:
                client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
                logger.info("Connected to remote Qdrant instance at %s", qdrant_url)
            else:
                storage_path = os.environ.get("QDRANT_PATH", "./qdrant_storage")
                os.makedirs(storage_path, exist_ok=True)
                client = QdrantClient(path=storage_path)
                logger.info("Using local persistent Qdrant storage at '%s'", storage_path)
            return client, collection_name
        except Exception:
            logger.exception("Failed to initialize Qdrant client.")
            raise

    def _ensure_collection_and_index(self, file_paths: List[str]) -> None:
        try:
            exists = self.qdrant_client.collection_exists(self.collection_name)
        except Exception:
            logger.exception("Failed to check whether Qdrant collection '%s' exists.", self.collection_name)
            raise

        if exists:
            try:
                point_count = self.qdrant_client.count(self.collection_name, exact=True).count
            except Exception:
                logger.exception("Failed to count vectors in existing collection; assuming it is empty.")
                point_count = 0
            if point_count > 0:
                logger.info(
                    "Reusing existing Qdrant collection '%s' (%d vectors) — skipping re-embedding.",
                    self.collection_name,
                    point_count,
                )
                return
            logger.info("Collection '%s' exists but is empty; indexing documentation now.", self.collection_name)
        else:
            vector_size = 384
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(
                    "Created new Qdrant collection '%s' (dim=%d, distance=COSINE).",
                    self.collection_name,
                    vector_size,
                )
            except Exception:
                logger.exception("Failed to create Qdrant collection '%s'.", self.collection_name)
                raise

        documents = self._load_documents(file_paths)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        if not docs:
            logger.warning("No documents were loaded — the knowledge base will be empty.")
            return

        try:
            QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embed_model,
            ).add_documents(docs)
            logger.info("Indexed %d chunks into Qdrant collection '%s'.", len(docs), self.collection_name)
        except Exception:
            logger.exception("Failed to index documents into Qdrant.")
            raise

    @staticmethod
    def _load_documents(file_paths: List[str]) -> List[Document]:
        documents: List[Document] = []
        for file_path in file_paths:
            try:
                if file_path.endswith(".csv"):
                    loader = CSVLoader(file_path=file_path)
                elif file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path=file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
                documents.extend(loader.load())
            except Exception:
                logger.exception("Failed to load document '%s' — skipping it.", file_path)
        return documents

    # ------------------------------------------------------------------------------------
    # Session memory (with summarization)
    # ------------------------------------------------------------------------------------
    def _get_session_state(self, session_id: str) -> SessionState:
        if session_id not in self.store:
            self.store[session_id] = SessionState()
        return self.store[session_id]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return self._get_session_state(session_id).history

    def hydrate_session(self, session_id: str, turns: List[Tuple[str, str]]) -> None:
        """Rebuilds in-memory state from persisted (role, content) turns (e.g. after a restart)."""
        if session_id in self.store:
            return
        state = self._get_session_state(session_id)
        for role, content in turns:
            if role == "user":
                state.history.add_user_message(content)
            else:
                state.history.add_ai_message(content)
        self._maybe_summarize(session_id)

    def forget_session(self, session_id: str) -> None:
        self.store.pop(session_id, None)

    def _get_effective_history(self, session_id: str) -> List[BaseMessage]:
        """Returns summary + recent turns only, bounding tokens sent to the LLM."""
        state = self._get_session_state(session_id)
        messages: List[BaseMessage] = list(state.history.messages)
        if state.summary:
            messages = [SystemMessage(content=f"Summary of earlier conversation: {state.summary}")] + messages
        return messages

    def _maybe_summarize(self, session_id: str) -> None:
        state = self._get_session_state(session_id)
        if len(state.history.messages) <= MAX_HISTORY_MESSAGES:
            return

        to_summarize = state.history.messages[:-SUMMARY_KEEP_RECENT]
        recent = state.history.messages[-SUMMARY_KEEP_RECENT:]
        conversation_text = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)

        try:
            new_summary = self.summarize_chain.invoke(
                {"previous_summary": state.summary or "None.", "conversation": conversation_text}
            ).strip()
            state.summary = new_summary
            state.history.clear()
            for m in recent:
                state.history.add_message(m)
            logger.info("[Memory] Summarized older turns for session %s (kept %d recent).", session_id, len(recent))
        except Exception:
            logger.exception("Memory summarization failed; keeping full history for session %s.", session_id)

    # ------------------------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _score_retrieval(self, question: str) -> Tuple[List[Document], float]:
        try:
            results = self.vector_store.similarity_search_with_score(question, k=6)
        except Exception:
            logger.exception("Qdrant similarity search failed; treating as no relevant context.")
            return [], 0.0
        docs = [doc for doc, _ in results]
        top_score = max((score for _, score in results), default=0.0)
        return docs, top_score

    def _cached_score_retrieval(self, question: str) -> Tuple[List[Document], float]:
        key = " ".join(question.strip().lower().split())
        if key in self._retrieval_cache:
            self._retrieval_cache.move_to_end(key)
            return self._retrieval_cache[key]

        result = self._score_retrieval(question)
        self._retrieval_cache[key] = result
        if len(self._retrieval_cache) > RETRIEVAL_CACHE_SIZE:
            self._retrieval_cache.popitem(last=False)
        return result

    def _contextualize_question(self, question: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return question
        try:
            standalone = self.contextualize_chain.invoke(
                {"input": question, "chat_history": chat_history}
            ).strip()
            return standalone or question
        except Exception:
            logger.exception("Contextualization failed; falling back to the original question.")
            return question

    def _evaluate_context(self, question: str, context: str) -> bool:
        try:
            verdict = self.evaluator_chain.invoke({"question": question, "context": context}).strip().upper()
            logger.info("[Evaluator] question=%r verdict=%s", question, verdict)
            return verdict.startswith("ENOUGH")
        except Exception:
            logger.exception("Context evaluation failed; treating context as insufficient.")
            return False

    def _rewrite_query(self, question: str) -> str:
        try:
            rewritten = self.rewrite_chain.invoke({"question": question}).strip()
            logger.info("[Rewrite] '%s' -> '%s'", question, rewritten)
            return rewritten or question
        except Exception:
            logger.exception("Query rewrite failed; retrying with the original question.")
            return question

    def _ask_clarification(self, question: str) -> str:
        try:
            return self.clarification_chain.invoke({"question": question}).strip()
        except Exception:
            logger.exception("Clarification generation failed; using a generic fallback.")
            return "Could you tell me a bit more about what you'd like help with?"

    def _retrieve_with_self_correction(
        self, question: str, seed_docs: Optional[List[Document]] = None
    ) -> Tuple[List[Document], bool]:
        current_question = question
        docs = seed_docs if seed_docs is not None else self.retriever.invoke(current_question)

        for attempt in range(MAX_RETRIES + 1):
            context = self._format_docs(docs)
            if self._evaluate_context(current_question, context):
                return docs, True
            if attempt < MAX_RETRIES:
                logger.info("[Self-Correction] Attempt %d insufficient, rewriting query.", attempt + 1)
                current_question = self._rewrite_query(current_question)
                docs = self.retriever.invoke(current_question)

        logger.info("[Self-Correction] Exhausted %d retries without sufficient context.", MAX_RETRIES)
        return docs, False

    # ------------------------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------------------------
    def qa_with_memory(self, user_question: str, session_id: str) -> List:
        logger.info("New query [session=%s]: %s", session_id, user_question)
        state = self._get_session_state(session_id)
        history = state.history

        try:
            effective_history = self._get_effective_history(session_id)
            standalone_question = self._contextualize_question(user_question, effective_history)

            docs, top_score = self._cached_score_retrieval(standalone_question)
            use_rag = bool(docs) and top_score >= RAG_SIMILARITY_THRESHOLD

            if use_rag:
                logger.info(
                    "[Route] mode=RAG session=%s top_score=%.4f threshold=%.2f",
                    session_id, top_score, RAG_SIMILARITY_THRESHOLD,
                )
                docs, sufficient = self._retrieve_with_self_correction(standalone_question, seed_docs=docs)

                if not sufficient:
                    clarification = self._ask_clarification(user_question)
                    history.add_user_message(user_question)
                    history.add_ai_message(clarification)
                    self._maybe_summarize(session_id)
                    return [clarification, [m.content for m in history.messages]]

                answer = self.answer_chain.invoke(
                    {"input": standalone_question, "context": docs, "chat_history": effective_history}
                )
            else:
                logger.info(
                    "[Route] mode=GENERAL_LLM session=%s top_score=%.4f threshold=%.2f",
                    session_id, top_score, RAG_SIMILARITY_THRESHOLD,
                )
                answer = self.general_chain.invoke(
                    {"input": standalone_question, "chat_history": effective_history}
                )

            history.add_user_message(user_question)
            history.add_ai_message(answer)
            self._maybe_summarize(session_id)
            return [answer, [m.content for m in history.messages]]

        except Exception:
            logger.exception("Unhandled error while answering query for session %s", session_id)
            fallback = "Sorry, something went wrong on my end. Could you please repeat that?"
            return [fallback, [m.content for m in history.messages]]