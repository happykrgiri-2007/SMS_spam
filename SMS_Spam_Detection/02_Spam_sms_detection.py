
# ============================================================
# PROJECT: SPAM SMS DETECTION
# INTERNSHIP: SAM AI TECHNOLOGIES
# ALGORITHM: TF-IDF + MULTINOMIAL NAIVE BAYES
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATASET_PATH = "spam.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "spam.csv not found. Place it in the project folder "
        "alongside spam_sms_detection.py."
    )

df = pd.read_csv(DATASET_PATH, encoding="latin-1")

print("=" * 60)
print("SPAM SMS DETECTION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)
print("\nDataset columns:", df.columns.tolist())

# ------------------------------------------------------------
# 2. PREPARE AND CLEAN DATA
# ------------------------------------------------------------

# This CSV commonly uses v1 for labels and v2 for messages.
if not {"v1", "v2"}.issubset(df.columns):
    raise ValueError(
        "Expected columns 'v1' and 'v2' were not found. "
        "Check the CSV file format."
    )

df["label"] = df["v1"].astype(str).str.strip().str.lower()

# Keep the main message and any extra fragments.
# Extra columns can occur in this dataset because some rows
# contain commas or inconsistent CSV quoting.
message_columns = [
    col for col in df.columns
    if col not in ["v1", "label"]
]

def combine_message_parts(row):
    parts = []
    for value in row:
        if pd.notna(value):
            part = str(value).strip()
            if part and part.lower() != "nan":
                parts.append(part)
    return " ".join(parts)

df["message"] = df[message_columns].apply(
    combine_message_parts,
    axis=1
)

# Keep only valid labels and non-empty messages.
df = df[df["label"].isin(["ham", "spam"])].copy()
df = df[df["message"].str.strip().ne("")].copy()

print("\nDataset shape after cleaning:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# Remove duplicate label-message pairs.
df = df.drop_duplicates(subset=["label", "message"]).copy()

print("\nDataset shape after removing duplicates:", df.shape)

# ------------------------------------------------------------
# 3. VISUALIZE DATASET
# ------------------------------------------------------------

df["label"].value_counts().reindex(
    ["ham", "spam"], fill_value=0
).plot(
    kind="bar",
    title="Distribution of Ham and Spam Messages",
    xlabel="Message Category",
    ylabel="Number of Messages"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df["message"]
y = df["label"]

# Split into 80% training and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 5. TF-IDF AND NAIVE BAYES MODEL
# ------------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 1)
        )
    ),
    ("classifier", MultinomialNB())
])

# Train the model using training data only.
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        target_names=["Ham (Legitimate)", "Spam"],
        zero_division=0
    )
)

# Create and save confusion matrix.
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Spam SMS Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 7. TEST CUSTOM SMS MESSAGES
# ------------------------------------------------------------

test_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hey bro, are we meeting today?",
    "URGENT! You have won a cash reward. Call now!",
    "Please call me when you reach home."
]

print("\n" + "=" * 60)
print("CUSTOM MESSAGE PREDICTIONS")
print("=" * 60)

predictions = model.predict(test_messages)

for message, prediction in zip(test_messages, predictions):
    category = (
        "SPAM"
        if prediction == "spam"
        else "HAM (LEGITIMATE)"
    )

    print("\nMessage:", message)
    print("Prediction:", category)

# ------------------------------------------------------------
# 8. SAVE TRAINED MODEL
# ------------------------------------------------------------

joblib.dump(model, "spam_sms_model.pkl")

print("\nTrained model saved as spam_sms_model.pkl")

# ------------------------------------------------------------
# 9. PREDICT A NEW MESSAGE ENTERED BY THE USER
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRY YOUR OWN SMS")
print("=" * 60)

user_message = input(
    "\nEnter an SMS message to check (or press Enter to skip): "
).strip()

if user_message:
    result = model.predict([user_message])[0]

    if result == "spam":
        print("\nResult: SPAM MESSAGE")
    else:
        print("\nResult: HAM (LEGITIMATE MESSAGE)")

print("\nProject execution completed!")

