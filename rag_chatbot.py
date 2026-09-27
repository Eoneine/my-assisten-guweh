"""
Personal Assistant Difa - RAG Chatbot (LangChain + Groq + ChromaDB)

This chatbot answers questions about Difa Leroy Gladion's work experience, projects, and
background, based on two knowledge sources:

    1. cv.md   -> history of experience, projects, education, skills
    2. faq.md  -> narrative answers for common interview questions
                  (motivation, strengths, work process, etc.),
                  which are not explicitly written in the CV

Both files are stored as structured markdown ("##" heading per
section, and "**Title**" line per entry within the Experience/
Project sections) so they can be chunked precisely: one chunk = one complete
unit of information, instead of character-count chunks that might cut
in the middle of a work experience.

How to run:
    python rag_chatbot.py

Prerequisites:
    - .env file containing GROQ_API_KEY in the same folder as this script
    - knowledge_docs/ folder containing cv.md and faq.md
    - Dependencies in requirements.txt installed
"""

import os
import re
import glob

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "asisten_difa"
TOP_K = 4

# All "##" sections containing a list of entries (jobs/projects)
# need to be split further per entry, so a single chunk does not contain multiple
# jobs/projects at once. Other sections (Professional Summary,
# Education, Certifications, Technical Skills & Tools) can be kept as a single
# complete chunk because their contents are brief and not a list of entries.
LIST_ENTRY_SECTIONS = {
    "Experience",
    "Organizational & Project Experience",
}

KNOWLEDGE_DIR = "./knowledge_docs"
SYSTEM_PROMPT_PATH = "./system_prompt.md"


# ============================================================
# 2. MODEL SETUP
# ============================================================


def initialize_llm() -> ChatGroq:
    """Prepare connection to the chat model via Groq."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=0.5,
        reasoning_effort="medium",
    )


# ============================================================
# 3. DATA INGESTION (Load -> Split -> Embed -> Store)
# ============================================================


def load_documents(folder: str) -> list[Document]:
    """
    Load all .md files in the folder into a list of Documents.

    Unlike PDF files which require a specific loader, markdown files can simply
    be read as plain text. The file name is stored in the "source" metadata
    so that later we can trace which chunk came from which file.
    """
    documents = []
    file_paths = sorted(glob.glob(os.path.join(folder, "*.md")))

    if not file_paths:
        raise FileNotFoundError(
            f"No .md files found in the '{folder}' folder. "
            "Ensure the knowledge_docs/ folder contains cv.md and faq.md."
        )

    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        documents.append(
            Document(page_content=content, metadata={"source": os.path.basename(path)})
        )

    return documents


def split_by_entry(section_text: str) -> list[str]:
    """
    Split the contents of a section into multiple chunks, one chunk per entry.

    In cv.md, every job/project entry starts with a line formatted as
    "**Company/Project Name** - ...". This line is used as a
    marker for "new entry starts here", so the splitting
    follows the original document structure, rather than character counts.
    """
    lines = section_text.split("\n")
    entries = []
    current_entry = []
    first_entry_found = False

    for line in lines:
        clean_line = line.strip()
        is_new_entry = bool(re.match(r"^\*\*.+\*\*", clean_line))

        if is_new_entry:
            first_entry_found = True
            if current_entry:
                entries.append("\n".join(current_entry).strip())
            current_entry = [line]
        elif first_entry_found:
            # The section heading line ("## Experience") before the first entry
            # is intentionally discarded, it doesn't need to be its own chunk.
            current_entry.append(line)

    if current_entry:
        entries.append("\n".join(current_entry).strip())

    return [entry for entry in entries if entry]


def split_markdown_documents(documents: list[Document]) -> list[Document]:
    """
    Split markdown documents into chunks, following the original heading and
    entry structure (rather than character-count splitting).

    Steps:
    1. Split each document by "##" headings using
       MarkdownHeaderTextSplitter -> one chunk per section.
    2. For sections containing a list of entries (Experience, Organizational
       & Project Experience), split further per entry via split_by_entry().
    3. Other sections (including the entirety of faq.md, which is already
       split per question because each question uses a "##" heading)
       are left as one chunk per section/question.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")],
        strip_headers=False,
    )

    final_chunks = []

    for doc in documents:
        sections = header_splitter.split_text(doc.page_content)

        for section in sections:
            section_name = section.metadata.get("section", "").strip()
            combined_metadata = {**doc.metadata, **section.metadata}

            if section_name in LIST_ENTRY_SECTIONS:
                for entry in split_by_entry(section.page_content):
                    final_chunks.append(
                        Document(page_content=entry, metadata=combined_metadata)
                    )
            else:
                final_chunks.append(
                    Document(
                        page_content=section.page_content, metadata=combined_metadata
                    )
                )

    return final_chunks


def build_vectorstore(documents: list[Document]) -> Chroma:
    """
    Convert a collection of Documents into a vector store ready for search.

    Embedding uses the default embedding function provided by ChromaDB
    (a lightweight ONNX model, already included in the chromadb dependency),
    so no additional packages like sentence-transformers are needed.
    """
    chunks = split_markdown_documents(documents)

    vectorstore = Chroma(collection_name=COLLECTION_NAME)
    # reset_collection() is used so this can safely run repeatedly without
    # accumulating old data every time the container is restarted.
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    return vectorstore


# ============================================================
# 4. RAG CHAIN (Context Injection -> Prompt -> Model -> Parser)
# ============================================================


def format_documents(document_list: list[Document]) -> str:
    """Combine multiple chunks from retrieval into a single context text."""
    return "\n\n".join(doc.page_content for doc in document_list)


def load_system_prompt(path: str) -> str:
    """Read the system prompt from a separate .md file."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_rag_chain(retriever, model: ChatGroq, system_prompt: str):
    """Combine retriever, prompt, model, and parser into a single LCEL chain."""
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    rag_chain = (
        RunnableParallel(
            context=retriever | format_documents,
            question=RunnablePassthrough(),
        )
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


# ============================================================
# 5. MAIN PROGRAM (terminal mode, for quick testing without UI)
# ============================================================


def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not found. Ensure the .env file exists in the same "
            "folder as this script and contains GROQ_API_KEY=..."
        )

    print("Preparing the model...")
    model = initialize_llm()

    print(f"Loading documents from folder '{KNOWLEDGE_DIR}'...")
    documents = load_documents(KNOWLEDGE_DIR)
    print(f"  -> {len(documents)} files successfully loaded.")

    print("Building vector store from documents...")
    vectorstore = build_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Loading system prompt from '{SYSTEM_PROMPT_PATH}'...")
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)

    print("Assembling RAG chain...")
    rag_chain = build_rag_chain(retriever, model, system_prompt)

    print("\nAssistant is ready. Type a question, or 'keluar' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"keluar", "exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        answer = rag_chain.invoke(question)
        print(f"Answer : {answer}\n")


if __name__ == "__main__":
    main()
