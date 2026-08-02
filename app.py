import hashlib
import io
import os

import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="PDF RAG App", layout="centered")
st.title("PDF Question Answering App")
st.write(
    "Upload a PDF, wait for the vector database to finish building, "
    "and then ask questions about the document."
)

# Create state variables once per browser session.
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


def build_vector_store(pdf_bytes: bytes):
    """Read PDF bytes, split readable text, and create a FAISS vector store."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number},
                )
            )

    if not page_documents:
        raise ValueError(
            "No readable text was found. The PDF may be scanned, image-only, or empty."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, len(reader.pages), len(chunks)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OPENAI_API_KEY is not set. Stop the app, export the key in the VS Code "
        "terminal, and run the app again."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload one PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to begin.")
else:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Rebuild only when a different PDF is uploaded.
    if st.session_state.file_hash != current_hash:
        st.session_state.vector_store = None
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0

        try:
            with st.spinner("Reading PDF and building vector database..."):
                vector_store, page_count, chunk_count = build_vector_store(pdf_bytes)

            st.session_state.vector_store = vector_store
            st.session_state.file_hash = current_hash
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count
        except Exception as error:
            st.session_state.file_hash = None
            st.error(f"Could not process this PDF: {error}")
            st.stop()

    if st.session_state.vector_store is not None:
        st.success(
            f"Ready: {st.session_state.pdf_name} | "
            f"{st.session_state.page_count} pages | "
            f"{st.session_state.chunk_count} chunks"
        )

        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question about the uploaded PDF",
                placeholder="Example: What are the key findings?",
            )
            submitted = st.form_submit_button("Ask")

        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching the PDF and generating an answer..."):
                    retrieved_docs = (
                        st.session_state.vector_store.similarity_search(
                            question,
                            k=4,
                        )
                    )

                    context = "\n\n".join(
                        f"[PDF page {doc.metadata.get('page', 'unknown')}]\n"
                        f"{doc.page_content}"
                        for doc in retrieved_docs
                    )

                    system_prompt = (
                        "Answer questions only from the supplied PDF context. "
                        "Do not use outside knowledge. If the answer is not present, say: "
                        "'I could not find that information in the uploaded PDF.' "
                        "Keep the answer clear and concise, and mention page numbers "
                        "when the context provides them."
                    )

                    user_prompt = (
                        f"Question:\n{question}\n\n"
                        f"PDF context:\n{context}"
                    )

                    llm = ChatOpenAI(model="gpt-4o-mini")
                    response = llm.invoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt),
                        ]
                    )

                st.subheader("Answer")
                answer_text = (
                    response.text
                    if hasattr(response, "text") and response.text
                    else response.content
                )
                st.write(answer_text)

                source_pages = sorted(
                    {
                        doc.metadata.get("page")
                        for doc in retrieved_docs
                        if doc.metadata.get("page") is not None
                    }
                )
                if source_pages:
                    st.caption(
                        "Retrieved PDF pages: "
                        + ", ".join(str(page) for page in source_pages)
                    )
