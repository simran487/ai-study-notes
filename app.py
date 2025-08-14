import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text


st.title("📚 AI Study Notes Generator (Free)")

uploaded_pdf = st.file_uploader("Upload your PDF lecture notes", type=["pdf"])


if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        text_data = extract_text_from_pdf(uploaded_pdf)

    if len(text_data.strip()) == 0:
        st.error("No readable text found in the PDF.")
    else:
        st.success("Text extracted successfully! Generating notes...")



    chunks = [text_data[i:i+1000] for i in range(0, len(text_data), 1000)]
    notes = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        notes.append(summary[0]['summary_text'])



    st.subheader("📌 AI-Generated Study Notes")
    st.write("\n\n".join(notes))

    # Download button
    notes_text = "\n\n".join(notes)
    st.download_button("Download Notes", notes_text, file_name="study_notes.txt")
