import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec

# ---------------------------
# Cache model loading
# ---------------------------
@st.cache_resource
def load_components():
    st.write("Loading model and Word2Vec… please wait")
    model = load_model("model.h5", compile=False)
    tag_encoder = pickle.load(open("tag_encoder.pkl", "rb"))
    w2v_model = Word2Vec.load("word2vec.model")
    return model, tag_encoder, w2v_model

model, tag_encoder, w2v_model = load_components()
w2v_kv = w2v_model.wv

# ---------------------------
# Constants
# ---------------------------
EMBED_DIM = 100
MAX_LEN = 50  # يجب أن يكون نفس طول الجملة أثناء التدريب

# ---------------------------
# Utility functions
# ---------------------------
def clean_word(w):
    w = w.strip().lower()
    w = re.sub(r"[^a-z0-9\-_]", "", w)
    return w

def sentence_to_w2v(sentence, kv, max_len=MAX_LEN, dim=EMBED_DIM):
    words = [clean_word(w) for w in sentence.split()]
    vectors = [kv[w] if w in kv else np.zeros(dim) for w in words]
    padded = np.zeros((1, max_len, dim))
    padded[0, :len(vectors)] = vectors[:max_len]
    return words, padded, min(len(vectors), max_len)

def extract_entities(words, tags):
    entities = []
    cur = []
    etype = None
    for w, t in zip(words, tags):
        if t.startswith("B-"):
            if cur:
                entities.append((etype, " ".join(cur)))
            etype = t.split("-")[1]
            cur = [w]
        elif t.startswith("I-") and etype:
            cur.append(w)
        else:
            if cur:
                entities.append((etype, " ".join(cur)))
                cur = []
                etype = None
    if cur:
        entities.append((etype, " ".join(cur)))
    return entities

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔍 Named Entity Recognition (NER) using BiLSTM + Word2Vec")
st.write("Enter a sentence below and click Predict:")

sentence = st.text_input("Enter a sentence:")

if st.button("Predict"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        words, x_vec, real_len = sentence_to_w2v(sentence, w2v_kv)
        pred = model.predict(x_vec)
        pred_ids = np.argmax(pred, axis=-1)[0][:real_len]
        tags = tag_encoder.inverse_transform(pred_ids)

        st.subheader("Token-level Tags:")
        for w, t in zip(words, tags):
            st.write(f"**{w}** → {t}")

        st.subheader("Extracted Entities:")
        ents = extract_entities(words, tags)
        if not ents:
            st.write("No entities found.")
        else:
            for et, val in ents:
                st.write(f"- **{et}** → {val}")
