import streamlit as st
import pypdf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Free PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 Free PDF RAG Lab")
st.write("Upload a PDF, build a searchable index, then ask questions about its content.")
st.info(
    "This demo uses free, locally loaded Hugging Face models. No paid API key is required. "
    "Free hosting and model execution still have CPU, memory, and usage limits; no hosted model is truly unlimited."
)

@st.cache_resource(show_spinner="Loading the free answer model (first run may take a few minutes)...")
def load_model():
    # Small instruction-tuned model; runs locally, without an API key.
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def extract_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages

def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_num, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            part = text[start:start + chunk_size]
            if part.strip():
                chunks.append({"page": page_num, "text": part})
            start += max(1, chunk_size - overlap)
    return chunks

uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded:
    try:
        pages = extract_pdf(uploaded)
        chunks = make_chunks(pages)
        if not chunks:
            st.error("No selectable text was found. If this is a scanned PDF, run OCR on it first.")
            st.stop()

        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)

        st.success(f"PDF read successfully: {len(pages)} text page(s), {len(chunks)} searchable chunk(s).")
        st.caption("The index is built in memory for this session. Uploading a different PDF rebuilds it.")

        question = st.text_input("Ask a question about your PDF", placeholder="What are the main points?")
        top_k = st.slider("Number of relevant text chunks", min_value=1, max_value=min(5, len(chunks)), value=min(3, len(chunks)))

        if st.button("Get answer", type="primary", disabled=not question.strip()):
            q_vec = vectorizer.transform([question])
            scores = cosine_similarity(q_vec, matrix).ravel()
            best = np.argsort(scores)[::-1][:top_k]
            context_parts = []
            for idx in best:
                if scores[idx] > 0:
                    context_parts.append(f"[Page {chunks[idx]['page']}] {chunks[idx]['text']}")
            if not context_parts:
                st.warning("I couldn't find a close text match. Try using words that appear in the PDF.")
            else:
                context = "\n".join(context_parts)
                prompt = (
                    "Answer the question using only the context below. "
                    "If the context does not contain the answer, say you cannot find it in the PDF.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                )
                with st.spinner("Searching the PDF and generating an answer..."):
                    model = load_model()
                    result = model(prompt, max_new_tokens=180, do_sample=False, truncation=True)
                st.subheader("Answer")
                st.write(result[0]["generated_text"])
                with st.expander("Retrieved source text"):
                    for idx in best:
                        if scores[idx] > 0:
                            st.markdown(f"**Page {chunks[idx]['page']} · Relevance score {scores[idx]:.3f}**")
                            st.write(chunks[idx]["text"])
                            st.divider()
    except Exception as e:
        st.error(f"Could not process this PDF: {e}")
        st.caption("Try a text-based PDF. Password-protected or scanned PDFs may need extra processing.")
else:
    st.markdown("### How it works")
    st.markdown("1. **Load:** extract text from your PDF.\n2. **Chunk:** split text into smaller pieces.\n3. **Retrieve:** find relevant chunks with TF-IDF similarity.\n4. **Generate:** pass those chunks and your question to the free FLAN-T5-small model.")
