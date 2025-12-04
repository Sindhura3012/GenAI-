import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

st.title("GenAI App with PDF/Text Input")

# -----------------------------------
# INPUT SECTION
# -----------------------------------
input_choice = st.radio("Select Input Type:", ["Upload PDF", "Write Text"])

text = ""

if input_choice == "Upload PDF":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        st.write("### Extracted Text:")
        st.write(text if text else "**No readable text found in PDF.**")

else:
    text = st.text_area("Enter your text:")

# -----------------------------------
# TASK SELECTION
# -----------------------------------
task = st.selectbox(
    "Choose a Task",
    [
        "Generate Text (GPT2)",
        "Summarize Text",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Translation (English → French)",
        "Paraphrasing",
        "Grammar Correction",
        "Sentence Similarity",
    ]
)

# -----------------------------------
# MODELS (Loaded Once)
# -----------------------------------
@st.cache_resource
def load_models():
    return {
        "gen": pipeline("text-generation", model="gpt2"),
        "sum": pipeline("summarization", model="facebook/bart-large-cnn"),
        "sent": pipeline("sentiment-analysis"),
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
        "trans": pipeline("translation", model="helsinki-NLP/opus-mt-en-fr"),
        "para": pipeline("text2text-generation", model="t5-small"),
        "gc": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
        "embed": SentenceTransformer("all-MiniLM-L6-v2")
    }

models = load_models()

# -----------------------------------
# RUN TASK
# -----------------------------------
st.write("### Output:")

if text.strip():

    if task == "Generate Text (GPT2)":
        output = models["gen"](text, max_length=300)[0]["generated_text"]
        st.write(output)

    elif task == "Summarize Text":
        output = models["sum"](text, max_length=120, min_length=40)[0]["summary_text"]
        st.write(output)

    elif task == "Sentiment Analysis":
        output = models["sent"](text)
        st.json(output)

    elif task == "Named Entity Recognition":
        output = models["ner"](text)
        st.json(output)

    elif task == "Translation (English → French)":
        output = models["trans"](text)[0]["translation_text"]
        st.write(output)

    elif task == "Paraphrasing":
        output = models["para"]("paraphrase: " + text)[0]["generated_text"]
        st.write(output)

    elif task == "Grammar Correction":
        output = models["gc"](text)[0]["generated_text"]
        st.write(output)

    elif task == "Sentence Similarity":
        text2 = st.text_input("Enter second sentence:")
        if text2:
            a = models["embed"].encode(text, convert_to_tensor=True)
            b = models["embed"].encode(text2, convert_to_tensor=True)
            score = util.pytorch_cos_sim(a, b).item()
            st.write(f"Similarity Score: {score}")
        else:
            st.write("Enter second sentence to calculate similarity.")

else:
    st.info("Upload a PDF or enter text above to see output.")
