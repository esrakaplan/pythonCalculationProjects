import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.DataFrame({
    "text": [
        "This product is amazing",
        "I love this phone",
        "Best purchase ever",
        "Absolutely fantastic experience",
        "Very happy with this",
        "Terrible quality",
        "Worst product ever",
        "I hate it",
        "Very disappointing",
        "Not worth the money",
        "Worst item ever"
    ],
    "label": [1,1,1,1,1,0,0,0,0,0,0]  # 1=Positive, 0=Negative
})

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ===== Text -> Numbers (TF-IDF) =====
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== Model =====
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ===== Prediction =====
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


new_comments = ["I really love this", "This is the worst thing I bought"]
new_vec = vectorizer.transform(new_comments)
predictions = model.predict(new_vec)

print("\nNew Predictions:")
for text, pred in zip(new_comments, predictions):
    print(text, "->", "Positive" if pred == 1 else "Negative")