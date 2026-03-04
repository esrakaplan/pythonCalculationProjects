import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== Mini Dataset =====
data = pd.DataFrame({
    "text": [
        "Win a free iPhone now",
        "Congratulations you won money",
        "Limited offer click now",
        "Earn dollars fast",
        "Meeting at 3 pm tomorrow",
        "Project deadline is next week",
        "Can we reschedule the meeting?",
        "Lunch at 12?",
        "Free vacation offer just for you",
        "Cheap loans available now"
    ],
    "label": [1,1,1,1,0,0,0,0,1,1]  # 1 = Spam, 0 = Not Spam
})

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.3, random_state=42
)

# ===== Text -> Numbers =====
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== Model =====
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Spam Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_emails = ["You won a free ticket", "Let's meet tomorrow"]
new_vec = vectorizer.transform(new_emails)
predictions = model.predict(new_vec)

print("\nNew Predictions:")
for text, pred in zip(new_emails, predictions):
    print(text, "->", "Spam" if pred == 1 else "Not Spam")