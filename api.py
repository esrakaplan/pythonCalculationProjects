from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Machine learning is a field of AI where models learn from data.",
    "Supervised learning uses labeled data.",
    "Unsupervised learning finds hidden patterns.",
    "Overfitting happens when model memorizes training data.",
    "Deep learning uses neural networks."
]

doc_embeddings = model.encode(documents)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    question_embedding = model.encode([query.question])
    similarities = cosine_similarity(question_embedding, doc_embeddings)
    best_index = np.argmax(similarities)

    return {
        "question": query.question,
        "best_match": documents[best_index],
        "similarity_score": float(similarities[0][best_index])
    }



"""
uvicorn api:app --reload


postman request POST 'http://127.0.0.1:8000/ask' \
  --header 'Content-Type: application/json' \
  --body '{
  "question": "What is supervised learning?"
}'"""