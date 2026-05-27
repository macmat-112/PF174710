# 2. Implementacja modeli:
# Użyjemy danych z poprzedniego pliku (dane.py).
# Zaimplementujemy dwa podejścia analizy sentymentu:
# - Podejście klasyczne: TF-IDF + Naive Bayes / Linear SVC / Logistic Regression.
# Jako że nasze zbiory są niezbalansowane, sprawdzimy także wyniki dla SVM i LR z zaaplikowanymi wagami.
# - Podejście Transformer: Fine-tuning modelu HerBERT (polski BERT), także z dodanymi wagami: 1 - neutralny, 10 - obraźliwy.

# Import bibliotek
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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from torch.nn import CrossEntropyLoss

# Ustawienia wyświetlania
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)

# Wczytaj dane z pliku "dane.py"
df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Przygotowanie danych dla podejścia klasycznego
X_train_processed = df_train["text_processed"].fillna("").values
X_val_processed = df_val["text_processed"].fillna("").values
X_test_processed = df_test["text_processed"].fillna("").values

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

X_train_tfidf = tfidf.fit_transform(X_train_processed)
X_val_tfidf = tfidf.transform(X_val_processed)
X_test_tfidf = tfidf.transform(X_test_processed)

print(f"Wymiary macierzy TF-IDF (train): {X_train_tfidf.shape}")
print(f"Liczba cech (unikalne n-gramy): {X_train_tfidf.shape[1]}")

# Trening klasyfikatorów klasycznych
classifiers = {
    "SVM (LinearSVC)": LinearSVC(max_iter=5000, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (LinearSVC) weighted": LinearSVC(max_iter=5000, random_state=42, class_weight="balanced"), # przetestujemy też ten sam model, ale z dodaną wagą dla kategorii
    "Logistic Regression weighted": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), # to, co wyżej
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
plt.savefig("modele_pomylki_klasyczne.png")

# Sprawdzenie GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Urządzenie: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Załadowanie tokenizatora i modelu HerBERT
MODEL_NAME = "allegro/herbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_safetensors=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, use_safetensors=True
).to(device)

print(f"Model: {MODEL_NAME}")
print(f"Parametry: {sum(p.numel() for p in model.parameters()):,}")

class SentimentDataset(Dataset):
    """Dataset dla recenzji z sentymentem."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Przygotowanie DataLoaderów (używamy surowego tekstu, nie przetworzonego)
MAX_LENGTH = 128
BATCH_SIZE = 16

train_dataset = SentimentDataset(
    df_train["text_clean"].values, df_train["target"].values, tokenizer, MAX_LENGTH
)
val_dataset = SentimentDataset(
    df_val["text_clean"].values, df_val["target"].values, tokenizer, MAX_LENGTH
)
test_dataset = SentimentDataset(
    df_test["text_clean"].values, df_test["target"].values, tokenizer, MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Batche treningowe: {len(train_loader)}")
print(f"Batche walidacyjne: {len(val_loader)}")

# Hiperparametry treningu
EPOCHS = 3
LEARNING_RATE = 2e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

def train_epoch(model, loader, optimizer, scheduler, device):
    """Jedna epoka treningu."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Chcemy zdefiniować wagi (hejt jest 10x ważniejszy) i wysłać na GPU
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        weights = torch.tensor([1.0, 10.0]).to(device) 
        loss_fct = CrossEntropyLoss(weight=weights)

        # Obliczamy loss ręcznie na podstawie logitów i prawdziwych etykiet
        loss = loss_fct(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Ewaluacja modelu."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)

# Pętla treningowa
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

print(f"Rozpoczynam trening HerBERT ({EPOCHS} epok)...\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(
        f"Epoka {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
    )

# Wykresy treningu
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(history["val_loss"], label="Val Loss", marker="o")
axes[0].set_title("Loss w trakcie treningu")
axes[0].set_xlabel("Epoka")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(history["val_acc"], label="Val Acc", marker="o")
axes[1].set_title("Accuracy w trakcie treningu")
axes[1].set_xlabel("Epoka")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("modele_wykresy_treningu.png")

# Ewaluacja HerBERT na zbiorze testowym
test_loss, test_acc, test_f1, y_test_pred_transformer, y_test_true = evaluate(
    model, test_loader, device
)

print(f"Wyniki HerBERT na zbiorze testowym:")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"\nRaport klasyfikacji:")
print(
    classification_report(
        y_test_true, y_test_pred_transformer,
        target_names=["Neutralny", "Obraźliwy"]
    )
)

# Macierz pomyłek - Transformer
cm_transformer = confusion_matrix(y_test_true, y_test_pred_transformer)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_transformer,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Neutralny", "Obraźliwy"],
    yticklabels=["Neutralny", "Obraźliwy"],
)
plt.title("Macierz pomyłek – HerBERT")
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")
plt.tight_layout()
plt.savefig("modele_pomylki_transformer.png")

# Zapis predykcji do pliku (do analizy w Notebooku 3)
results_df = df_test.copy()
results_df["pred_classic"] = y_test_pred_classic
results_df["pred_transformer"] = y_test_pred_transformer
results_df.to_csv("test_predictions.csv", index=False)

# Zapis podsumowania wyników
summary = {
    "classic_name": best_classic_name,
    "classic_accuracy": classic_test_acc,
    "classic_f1": classic_test_f1,
    "transformer_name": "HerBERT",
    "transformer_accuracy": test_acc,
    "transformer_f1": test_f1,
}

with open("results_summary.pkl", "wb") as f:
    pickle.dump(summary, f)

print("Predykcje zapisane do: test_predictions.csv")
print("Podsumowanie wyników zapisane do: results_summary.pkl")
