import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDYQeM1YSwLLRNAg2r71mUALYZbnDrZcjA"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# LOAD DATASET
# -----------------------
df = pd.read_excel("/Users/abishekr/Documents/chatbot-app/chatbot-app/backend/miniprojectdataset.xlsx")

texts = []
for _, row in df.iterrows():
    row_text = " | ".join(
        [f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])]
    )
    texts.append(row_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# -----------------------
# GEMINI MODEL
# -----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6
)

class Query(BaseModel):
    question: str

# -----------------------
# HUMAN CHAT DETECTOR
# -----------------------
def is_human_chat(text):
    text = text.lower().strip()

    casual_phrases = [
        "hi", "hello", "hey",
        "how are you",
        "who are you",
        "what is your name",
        "what's your name",
        "thank you", "thanks",
        "bye", "goodbye",
        "good morning",
        "good evening",
        "good night",
        "i need your help",
        "can you help me",
        "help me",
        "how can you help",
        "nice",
        "okay",
        "ok"
    ]

    for phrase in casual_phrases:
        if phrase in text:
            return True

    return False

# -----------------------
# DATASET SEARCH
# -----------------------
def search_dataset(query, threshold=0.18, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    best_score = max(scores)

    if best_score < threshold:
        return None

    top_indices = np.argsort(scores)[::-1][:top_k]
    context = "\n".join([texts[i] for i in top_indices])

    return context

# -----------------------
# API
# -----------------------
@app.post("/query")
def query_api(q: Query):
    question = q.question.strip()

    # Human chat
    if is_human_chat(question):
        prompt = f"""
You are a friendly chatbot.
Reply naturally and warmly.

User: {question}
"""
        response = llm.invoke(prompt)
        return {"answer": response.content}

    # Dataset search
    context = search_dataset(question)

    if context:
        prompt = f"""
You are an assistant.

Answer only using the dataset.

Dataset:
{context}

Question:
{question}

If answer is present, answer clearly.
If not present, say:
"I couldn't find exact information in the dataset."
"""
        response = llm.invoke(prompt)
        return {"answer": response.content}

    return {
        "answer": "I couldn't find that information in the dataset."
    }

@app.get("/")
def home():
    return {"message": "Backend running"}