# rag_system.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

# Mini knowledge base
documents = [
    "Machine learning is a field of AI where models learn from data.",
    "Supervised learning uses labeled data.",
    "Unsupervised learning finds hidden patterns.",
    "Overfitting happens when model memorizes training data.",
    "Deep learning uses neural networks."
]

# Document embeddings
doc_embeddings = model.encode(documents)


def retrieve_answer(question):
    question_embedding = model.encode([question])

    similarities = cosine_similarity(question_embedding, doc_embeddings)
    best_index = np.argmax(similarities)

    return {
        "question": question,
        "best_match": documents[best_index],
        "similarity_score": float(similarities[0][best_index])
    }


# Test
if __name__ == "__main__":
    result = retrieve_answer("Explain supervised models")
    print(result)
    result = retrieve_answer("Explain Machine learning")
    print(result)