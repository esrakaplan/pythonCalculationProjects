import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== Mini Knowledge Base =====
questions = [
    "What is machine learning?",
    "What is Python?",
    "What is overfitting?",
    "What is supervised learning?"
]

answers = [
    "Machine learning is a field of AI where models learn from data.",
    "Python is a popular programming language.",
    "Overfitting happens when a model memorizes training data.",
    "Supervised learning uses labeled data to train models."
]

# ===== Vectorizer =====
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def chatbot(user_question):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, question_vectors)
    best_match_index = np.argmax(similarities)
    return answers[best_match_index]

# ===== Test =====
print(chatbot("Explain supervised learning"))
print(chatbot("Tell me about Python"))
print(chatbot("What is overfitting in ML?"))