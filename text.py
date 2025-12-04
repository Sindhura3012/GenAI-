import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Title
st.title("GenAI App with Transformers")

# User Options
option = st.selectbox(
    "Choose a task",
    [
        "Generate Text (GPT2)",
        "Summarize Text",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Translation (English → French)",
        "Paraphrasing",
        "Grammar Correction",
        "Sentence Similarity"
    ]
)

# -------------------------------------------
# MODELS
# -------------------------------------------
gen = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
translate = pipeline("translation", model="helsinki-NLP/opus-mt-en-fr")
para = pipeline("text2text-generation", model="t5-small")
gc = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------
# USER INPUT
# -------------------------------------------
user_text = st.text_area("Enter your text here:")

# -------------------------------------------
# EXECUTE BASED ON OPTION
# -------------------------------------------
if st.button("Run"):

    # 1. Generate Text
    if option == "Generate Text (GPT2)":
        result = gen(user_text, max_length=300)[0]['generated_text']
        st.write(result)

    # 2. Summarize Text
    elif option == "Summarize Text":
        result = summarizer(user_text, max_length=120, min_length=40)[0]['summary_text']
        st.write(result)

    # 3. Sentiment Analysis
    elif option == "Sentiment Analysis":
        result = sentiment(user_text)
        st.write(result)

    # 4. Named Entity Recognition
    elif option == "Named Entity Recognition":
        result = ner(user_text)
        st.write(result)

    # 5. Translation
    elif option == "Translation (English → French)":
        result = translate(user_text)[0]["translation_text"]
        st.write(result)

    # 6. Paraphrasing
    elif option == "Paraphrasing":
        result = para("paraphrase: " + user_text)[0]["generated_text"]
        st.write(result)

    # 7. Grammar Correction
    elif option == "Grammar Correction":
        result = gc(user_text)[0]["generated_text"]
        st.write(result)

    # 8. Sentence Similarity
    elif option == "Sentence Similarity":
        st.write("Enter second sentence below:")
        user_text2 = st.text_input("Second sentence:")
        if user_text2:
            a = embed_model.encode(user_text, convert_to_tensor=True)
            b = embed_model.encode(user_text2, convert_to_tensor=True)
            score = util.pytorch_cos_sim(a, b).item()
            st.write(f"Similarity Score: {score}")
