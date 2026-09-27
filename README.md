# Ask Difa - Personal RAG Chatbot

A chatbot that answers questions about Difa Leroy Gladion's work experience, projects, and
background, built with RAG (Retrieval-Augmented
Generation) so its answers are always based on source documents, not
hallucinations from the model.

**Demo**: [insert Streamlit Cloud link here after deploy]

## Brief Architecture

```
knowledge_docs/*.md -> split per section/entry -> embedding (ChromaDB) -> vector store
                                                                              |
user question -> retriever (top-k) -> prompt + context -> Groq LLM -> answer
```

The source documents (`cv.md`, `faq.md`) are split following their original heading and
entry structure (instead of character-count splitting), so that one chunk contains
one complete work experience/project/question. Embedding uses the
default embedding function provided by ChromaDB (lightweight, without additional
dependencies like sentence-transformers).

## Tech Stack

- [LangChain](https://python.langchain.com/) - RAG chain orchestration (LCEL)
- [Groq](https://groq.com/) - LLM inference
- [ChromaDB](https://www.trychroma.com/) - vector store
- [Streamlit](https://streamlit.io/) - chat interface

## Running Locally

1. Clone this repo and enter the folder
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root folder, containing:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get the API key from [console.groq.com](https://console.groq.com/keys)
4. Run the terminal version (optional, to check retrieval before using UI):
   ```
   python rag_chatbot.py
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (the `chroma_db/` folder is automatically not committed
   because it's in `.gitignore` - this is intentional, the vector store is rebuilt
   automatically every time the application starts)
2. Create a new app at [share.streamlit.io](https://share.streamlit.io/),
   point it to `app.py`
3. In the **Secrets** menu, fill in:
   ```
   GROQ_API_KEY = "your_key_here"
   ```

## Folder Structure

```
.
├── .streamlit/config.toml   # color theme
├── assets/style.css         # custom chat bubble CSS
├── knowledge_docs/          # knowledge sources (cv.md, faq.md)
├── app.py                   # Streamlit interface
├── rag_chatbot.py           # RAG logic (loader, splitter, chain)
├── system_prompt.md         # persona instructions & answer rules
└── requirements.txt
```
