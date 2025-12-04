import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# -----------------------------
# Title
# -----------------------------
st.title("GenAI App with PDF/Text Input")

# -----------------------------
# Input Choice
# -----------------------------
input_choice = st.radio(
    "Select Input Type:",
    ["Upload PDF", "Write Text"]
)

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
text = ""

if input_choice == "Upload PDF":
    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

        st.subheader("Extracted Text:")
        st.write(text)

elif input_choice == "Write Text":
    text = st.text_area("Enter your text here:")


# -----------------------------
# Select GenAI Task
# -----------------------------
task = st.selectbox(
    "Choose a GenAI Task",
    [
        "Generate Text (GPT2)",
        "Summarize Text",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Translation (English → French)",
        "Paraphrasing",
        "Grammar Correction",
        "Sentence Similarity (enter 2nd sentence)",
    ]
)

# Load models
gen = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
translate = pipeline("translation", model="helsinki-NLP/opus-mt-en-fr")
para = pipeline("text2text-generation", model="t5-small")
gc = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Run Button
# -----------------------------
if st.button("Run"):

    if not text.strip():
        st.error("Please upload a PDF or enter some text.")
    else:

        # Generate Text
        if task == "Generate Text (GPT2)":
            result = gen(text, max_length=300)[0]["generated_text"]
            st.write(result)

        # Summarization
        elif task == "Summarize Text":
            result = summarizer(text, max_length=120, min_length=40)[0]["summary_text"]
            st.write(result)

        # Sentiment
        elif task == "Sentiment Analysis":
            result = sentiment(text)
            st.write(result)

        # NER
        elif task == "Named Entity Recognition":
            result = ner(text)
            st.write(result)

        # Translation
        elif task == "Translation (English → French)":
            result = translate(text)[0]["translation_text"]
            st.write(result)

        # Paraphrasing
        elif task == "Paraphrasing":
            result = para("paraphrase: " + text)[0]["generated_text"]
            st.write(result)

        # Grammar Correction
        elif task == "Grammar Correction":
            result = gc(text)[0]["generated_text"]
            st.write(result)

        # Sentence Similarity
        elif task == "Sentence Similarity (enter 2nd sentence)":
            text2 = st.text_input("Enter Second Sentence:")
            if text2:
                vec1 = embed_model.encode(text, convert_to_tensor=True)
                vec2 = embed_model.encode(text2, convert_to_tensor=True)
                score = util.pytorch_cos_sim(vec1, vec2).item()
                st.write(f"Similarity Score: {score}")
            else:
                st.warning("Please enter the second sentence.")
