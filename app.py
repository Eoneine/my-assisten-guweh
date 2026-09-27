import os

import streamlit as st
from dotenv import load_dotenv

from rag_chatbot import (
    KNOWLEDGE_DIR,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    initialize_llm,
    load_documents,
    build_vectorstore,
    load_system_prompt,
    build_rag_chain,
)


NAME = "Difa Leroy Gladion"
ASSISTANT_AVATAR = "assets/avatar.jpg"
CSS_PATH = "assets/style.css"

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=f"Ask {NAME.split()[0]}",
    page_icon=":material/psychology:",
)


def inject_custom_css(path: str):
    """Inject custom CSS file into the page, if the file exists."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


inject_custom_css(CSS_PATH)

# ============================================================
# 2. CHECK API KEY
# ============================================================

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Check the .env file (locally) "
        "or the Secrets menu (on Streamlit Cloud)."
    )
    st.stop()

# ============================================================
# 3. PREPARE CHATBOT ENGINE (once, then cached)
# ============================================================


@st.cache_resource(show_spinner="Preparing assistant, please wait...")
def setup_chatbot():
    model = initialize_llm()
    documents = load_documents(KNOWLEDGE_DIR)
    vectorstore = build_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    return build_rag_chain(retriever, model, system_prompt)


rag_chain = setup_chatbot()

# ============================================================
# 4. CONVERSATION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


def get_assistant_avatar():
    """Use custom photo if the file exists, otherwise fallback to emoji."""
    return ASSISTANT_AVATAR if os.path.exists(ASSISTANT_AVATAR) else "🤖"


# ============================================================
# 5. UI
# ============================================================

with st.sidebar:
    st.header(f"About {NAME}")
    st.write(
        f"This assistant answers questions regarding {NAME}'s work experience, "
        f"projects, and background, based on the CV and additional notes "
        f"provided."
    )
    st.caption(
        "Answers are solely derived from the source documents, not from the internet."
    )
    st.divider()
    st.markdown(
        "[GitHub](https://github.com/Eoneine) | "
        "[LinkedIn](https://www.linkedin.com/in/difaleroygladion/) | "
        "[Portfolio](https://difaleroy.notion.site/difaleroyportfolio)"
    )
    if st.button("Start new conversation"):
        st.session_state.history = []

st.title(f"Ask {NAME}")
st.caption("Ask anything about my work experience and projects.")

with st.chat_message("assistant", avatar=get_assistant_avatar()):
    st.markdown(
        "Hello, please ask a question. Examples: "
        '"Tell me about the last project you worked on" or '
        '"What are your main skills?"'
    )

for message in st.session_state.history:
    avatar = get_assistant_avatar() if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ============================================================
# 6. Q&A
# ============================================================

question = st.chat_input("Write your question here...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=get_assistant_avatar()):
        with st.spinner("Searching for answers in documents..."):
            answer = st.write_stream(rag_chain.stream(question))
    st.session_state.history.append({"role": "assistant", "content": answer})
