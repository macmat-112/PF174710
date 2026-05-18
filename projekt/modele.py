import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)

# Wczytaj dane z pliku "dane.py"
df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Przygotowanie danych dla podejścia klasycznego
X_train_clean = df_train["text_processed"].fillna("").values
X_val_clean = df_val["text_processed"].fillna("").values
X_test_clean = df_test["text_processed"].fillna("").values

y_train = df_train["target"].values
y_val = df_val["target"].values
y_test = df_test["target"].values

# Wektoryzacja TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # unigramy i bigramy
    min_df=2,
    max_df=0.95,
)

X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_val_tfidf = tfidf.transform(X_val_clean)
X_test_tfidf = tfidf.transform(X_test_clean)

print(f"Wymiary macierzy TF-IDF (train): {X_train_tfidf.shape}")
print(f"Liczba cech (unikalne n-gramy): {X_train_tfidf.shape[1]}")

# Trening klasyfikatorów klasycznych
classifiers = {
    "SVM (LinearSVC)": LinearSVC(max_iter=5000, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (LinearSVC) weighted": LinearSVC(max_iter=5000, random_state=42, class_weight="balanced"),
    "Logistic Regression weighted": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
}

classic_results = {}

for name, clf in classifiers.items():
    print(f"\n{'='*50}")
    print(f"Trening: {name}")
    print(f"{'='*50}")

    # Trening
    clf.fit(X_train_tfidf, y_train)

    # Predykcja na zbiorze walidacyjnym
    y_val_pred = clf.predict(X_val_tfidf)

    # Metryki
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro") # obliczając f1_score, nie patrzymy na to, że neutralnych jest znacznie więcej od obraźliwych.

    classic_results[name] = {
        "model": clf,
        "val_accuracy": acc,
        "val_f1": f1,
        "val_predictions": y_val_pred,
    }

    print(f"Accuracy (val): {acc:.4f}")
    print(f"F1-score (val): {f1:.4f}")
    print(f"\nRaport klasyfikacji (val):")
    print(
        classification_report(
            y_val, y_val_pred, target_names=["Neutralny", "Obraźliwy"]
        )
    )

# Wybór najlepszego klasyfikatora klasycznego
best_classic_name = max(classic_results, key=lambda x: classic_results[x]["val_f1"])
best_classic = classic_results[best_classic_name]

print(f"Najlepszy klasyfikator klasyczny: {best_classic_name}")
print(f"F1-score (val): {best_classic['val_f1']:.4f}")

# Ewaluacja na zbiorze testowym
y_test_pred_classic = best_classic["model"].predict(X_test_tfidf)
classic_test_acc = accuracy_score(y_test, y_test_pred_classic)
classic_test_f1 = f1_score(y_test, y_test_pred_classic, average="macro") # tak, jak wyżej.

print(f"\nWyniki na zbiorze testowym ({best_classic_name}):")
print(f"Accuracy: {classic_test_acc:.4f}")
print(f"F1-score: {classic_test_f1:.4f}")
print(f"\nRaport klasyfikacji (test):")
print(
    classification_report(
        y_test, y_test_pred_classic, target_names=["Neutralny", "Obraźliwy"]
    )
)

# Macierz pomyłek - podejście klasyczne
cm_classic = confusion_matrix(y_test, y_test_pred_classic)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_classic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Neutralny", "Obraźliwy"],
    yticklabels=["Neutralny", "Obraźliwy"],
)
plt.title(f"Macierz pomyłek – {best_classic_name}")
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")
plt.tight_layout()
plt.show()
