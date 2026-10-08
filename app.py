import os
import time
import pickle
import tomllib
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai


# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AeroAssist AI", page_icon="✈️", layout="wide")
st.title("✈️ AeroAssist AI")
st.caption("FAA Aviation Maintenance Handbook Assistant")
st.info(
    "Ask questions about aviation maintenance. "
    "Always verify maintenance decisions against approved documentation."
)


# 2. GEMINI API KEY
# Priority: environment variable, Streamlit secrets, local TOML file.
api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    try:
        api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        api_key = ""

if not api_key:
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    try:
        with secrets_path.open("rb") as secrets_file:
            secrets_data = tomllib.load(secrets_file)
        api_key = str(secrets_data.get("GEMINI_API_KEY", "")).strip()
    except FileNotFoundError:
        st.error(
            "Gemini API key not found. Expected GEMINI_API_KEY in the "
            "environment or in .streamlit/secrets.toml."
        )
        st.stop()
    except tomllib.TOMLDecodeError:
        st.error(
            "Invalid TOML in .streamlit/secrets.toml. The file should contain "
            'one line: GEMINI_API_KEY = "your-private-key"'
        )
        st.stop()
    except Exception as error:
        st.error(f"Could not read secrets.toml: {type(error).__name__}: {error}")
        st.stop()

if not api_key:
    st.error("GEMINI_API_KEY is empty. Check .streamlit/secrets.toml.")
    st.stop()


# 3. LOAD FAISS INDEX, METADATA AND EMBEDDING MODEL
PROJECT_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_resources():
    index_path = PROJECT_DIR / "aeroassist_index.faiss"
    metadata_path = PROJECT_DIR / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing file: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, embedding_model


# 4. INITIALIZE GEMINI AND LOAD RESOURCES
try:
    client = genai.Client(api_key=api_key)
    index, metadata, embedding_model = load_resources()
except Exception as error:
    st.error(f"Could not initialize AeroAssist: {error}")
    st.stop()

st.success(f"Knowledge base loaded: {index.ntotal:,} indexed vectors")


# 5. RETRIEVAL-AUGMENTED GENERATION
def answer_question(question, index, metadata, embedding_model, client):
    question_embedding = embedding_model.encode(
        [question], convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(question_embedding, 5)
    context_parts = []
    sources = []

    for position in indices[0]:
        if position < 0 or position >= len(metadata):
            continue

        chunk = metadata[position]
        if isinstance(chunk, dict):
            source = chunk.get("source", "Unknown source")
            page = chunk.get("page", "Unknown page")
            content = chunk.get("text", str(chunk))
        else:
            source = "Unknown source"
            page = "Unknown page"
            content = str(chunk)

        context_parts.append(f"Source: {source}, Page: {page}\n{content}")
        source_item = {"source": source, "page": page}
        if source_item not in sources:
            sources.append(source_item)

    if not context_parts:
        return (
            "No relevant handbook passages were retrieved. Try rephrasing your question.",
            [],
        )

    context = "\n\n".join(context_parts)
    prompt = f"""
You are AeroAssist AI, an assistant for aviation maintenance learning.

INSTRUCTIONS:
1. Answer using only the supplied handbook context.
2. Do not invent technical specifications, limits, maintenance procedures, or requirements.
3. If the context is insufficient, clearly say so.
4. Cite supporting information using the provided handbook filename and page number.
5. Do not claim that your answer replaces approved maintenance data.
6. Explain technical concepts clearly and accurately.

HANDBOOK CONTEXT:
{context}

QUESTION:
{question}

Provide a concise, technically grounded answer.
"""

    response = None
    for attempt in range(3):
        try:
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt,
            )
            break
        except Exception as error:
            error_text = str(error).lower()
            is_temporary = (
                "503" in error_text
                or "429" in error_text
                or "service_unavailable" in error_text
                or "resource_exhausted" in error_text
                or "high demand" in error_text
            )
            if not is_temporary or attempt == 2:
                raise

            wait_seconds = 2 ** (attempt + 2)
            st.warning(
                "Gemini is temporarily busy. "
                f"Retrying in {wait_seconds} seconds (attempt {attempt + 2} of 3)..."
            )
            time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError("Gemini did not return a response.")

    answer = response.output_text
    if not answer:
        answer = "Gemini returned no text. Please try again."
    return answer, sources


# 6. DISPLAY CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved handbook sources"):
                for item in message["sources"]:
                    st.write(f'{item["source"]} — page {item["page"]}')


# 7. ACCEPT USER QUESTIONS
question = st.chat_input("Ask an aviation maintenance question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching FAA handbooks and generating an answer..."):
                answer, sources = answer_question(
                    question, index, metadata, embedding_model, client
                )

            st.markdown(answer)
            if sources:
                with st.expander("Retrieved handbook sources"):
                    for item in sources:
                        st.write(f'{item["source"]} — page {item["page"]}')

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            st.error(f"Unable to generate an answer: {error}")
            st.caption(
                "Check your internet connection, API access, and the PowerShell terminal for details."
            )
