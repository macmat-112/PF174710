# 3. Ewaluacja i analiza wyników
# Korzystając z danych z poprzednich plików:
# - Porównamy ilościowo oba podejścia (klasyczne vs Transformer).
# - Przeprowadzimy analizę błędów.
# - Zidentyfikujemy typowe pomyłki modeli z przykładami.
# - Sformułujemy wnioski.

# Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Ustawienia wyświetlania
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

# Wczytanie predykcji z Notebooka 2
results_df = pd.read_csv("test_predictions.csv")

with open("results_summary.pkl", "rb") as f:
    summary = pickle.load(f)

TEXT_COL = "sentence"
LABEL_COL = "target"

y_true = results_df[LABEL_COL].values
y_pred_classic = results_df["pred_classic"].values
y_pred_transformer = results_df["pred_transformer"].values

print(f"Liczba próbek testowych: {len(results_df)}")
print(f"Podejście klasyczne: {summary['classic_name']}")
print(f"Podejście Transformer: {summary['transformer_name']}")

# Tabela porównawcza metryk
metrics_data = []

for name, y_pred in [
    (summary["classic_name"], y_pred_classic),
    ("HerBERT (Transformer)", y_pred_transformer),
]:
    metrics_data.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1-score": f1_score(y_true, y_pred, average="macro"),
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.set_index("Model")

print("Porównanie metryk na zbiorze testowym:")
print("=" * 60)
print(metrics_df.round(4).to_string())
print("=" * 60)

# Wykres porównawczy metryk
metrics_plot = metrics_df.T

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_plot.index))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    metrics_plot.iloc[:, 0],
    width,
    label=metrics_plot.columns[0],
    color="steelblue",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    metrics_plot.iloc[:, 1],
    width,
    label=metrics_plot.columns[1],
    color="coral",
    alpha=0.8,
)

ax.set_ylabel("Wartość metryki")
ax.set_title("Porównanie modeli – metryki na zbiorze testowym")
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot.index)
ax.legend()
ax.set_ylim(0, 1.05)

# Wartości nad słupkami
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig("ewal_metryki_porownawczy.png")

# Macierze pomyłek obok siebie
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, y_pred, cmap) in enumerate([
    (summary["classic_name"], y_pred_classic, "Blues"),
    ("HerBERT", y_pred_transformer, "Greens"),
]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        ax=axes[idx],
        xticklabels=["Neutralny", "Obraźliwy"],
        yticklabels=["Neutralny", "Obraźliwy"],
    )
    axes[idx].set_title(f"Macierz pomyłek – {name}")
    axes[idx].set_xlabel("Predykcja")
    axes[idx].set_ylabel("Prawdziwa etykieta")

plt.tight_layout()
plt.savefig("ewal_macierze_pomylek.png")

# Szczegółowe raporty klasyfikacji
print(f"{'='*60}")
print(f"Raport klasyfikacji – {summary['classic_name']}")
print(f"{'='*60}")
print(classification_report(
    y_true, y_pred_classic, target_names=["Neutralny", "Obraźliwy"]
))

print(f"\n{'='*60}")
print(f"Raport klasyfikacji – HerBERT")
print(f"{'='*60}")
print(classification_report(
    y_true, y_pred_transformer, target_names=["Neutralny", "Obraźliwy"]
))

# Identyfikacja błędnych predykcji
results_df["classic_correct"] = (y_true == y_pred_classic)
results_df["transformer_correct"] = (y_true == y_pred_transformer)

# Kategorie błędów
results_df["error_type"] = "oba poprawne"
results_df.loc[
    ~results_df["classic_correct"] & results_df["transformer_correct"],
    "error_type"
] = "tylko klasyczny błędny"
results_df.loc[
    results_df["classic_correct"] & ~results_df["transformer_correct"],
    "error_type"
] = "tylko transformer błędny"
results_df.loc[
    ~results_df["classic_correct"] & ~results_df["transformer_correct"],
    "error_type"
] = "oba błędne"

print("Kategorie błędów:")
error_counts = results_df["error_type"].value_counts()
print(error_counts)
print(f"\nProcent zgodności obu modeli: {(error_counts.get('oba poprawne', 0)) / len(results_df) * 100:.1f}%")

# Wykres kategorii błędów
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["mediumseagreen", "steelblue", "coral", "crimson"]
error_counts.plot(kind="bar", color=colors[:len(error_counts)], ax=ax)
ax.set_title("Kategorie błędów – porównanie modeli")
ax.set_ylabel("Liczba próbek")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("ewal_kat_bledow.png")

# Przykłady błędnych predykcji – podejście klasyczne
print("PRZYKŁADY BŁĘDNYCH PREDYKCJI – Podejście klasyczne")
print("=" * 70)

classic_errors = results_df[~results_df["classic_correct"]].head(5)

for _, row in classic_errors.iterrows():
    text = str(row[TEXT_COL])[:200]
    true_label = "Obraźliwy" if row[LABEL_COL] == 1 else "Neutralny"
    pred_label = "Obraźliwy" if row["pred_classic"] == 1 else "Neutralny"
    print(f"\nTekst: {text}...")
    print(f"  Prawda: {true_label} | Predykcja: {pred_label}")
    print("-" * 70)

# Przykłady błędnych predykcji – Transformer
print("PRZYKŁADY BŁĘDNYCH PREDYKCJI – HerBERT (Transformer)")
print("=" * 70)

transformer_errors = results_df[~results_df["transformer_correct"]].head(5)

for _, row in transformer_errors.iterrows():
    text = str(row[TEXT_COL])[:200]
    true_label = "Obraźliwy" if row[LABEL_COL] == 1 else "Neutralny"
    pred_label = "Obraźliwy" if row["pred_transformer"] == 1 else "Neutralny"
    print(f"\nTekst: {text}...")
    print(f"  Prawda: {true_label} | Predykcja: {pred_label}")
    print("-" * 70)

# Przykłady gdzie oba modele się mylą (najtrudniejsze przypadki)
print("NAJTRUDNIEJSZE PRZYPADKI – oba modele błędne")
print("=" * 70)

both_wrong = results_df[results_df["error_type"] == "oba błędne"].head(5)

if len(both_wrong) > 0:
    for _, row in both_wrong.iterrows():
        text = str(row[TEXT_COL])[:200]
        true_label = "Obraźliwy" if row[LABEL_COL] == 1 else "Neutralny"
        print(f"\nTekst: {text}...")
        print(f"  Prawdziwa etykieta: {true_label}")
        print("-" * 70)
else:
    print("Brak przypadków, gdzie oba modele się mylą.")

# Analiza błędów wg długości tekstu
results_df["word_count"] = results_df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))

# Binowanie długości
bins = [0, 10, 25, 50, 100, float("inf")]
labels = ["1-10", "11-25", "26-50", "51-100", "100+"]
results_df["length_bin"] = pd.cut(results_df["word_count"], bins=bins, labels=labels)

# Accuracy per bin
acc_by_length = results_df.groupby("length_bin", observed=False).agg(
    classic_acc=("classic_correct", "mean"),
    transformer_acc=("transformer_correct", "mean"),
    count=("classic_correct", "count"),
).round(4)

print("Accuracy wg długości tekstu (słowa):")
print(acc_by_length.to_string())

# Wykres
fig, ax = plt.subplots(figsize=(10, 5))
acc_by_length[["classic_acc", "transformer_acc"]].plot(
    kind="bar", ax=ax, color=["steelblue", "coral"]
)
ax.set_title("Accuracy wg długości recenzji")
ax.set_xlabel("Liczba słów")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.legend([summary["classic_name"], "HerBERT"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("ewal_accuracy.png")

# Przypadki gdzie Transformer jest lepszy od klasycznego
print("PRZEWAGA TRANSFORMERA – poprawny Transformer, błędny klasyczny")
print("=" * 70)

transformer_wins = results_df[
    results_df["error_type"] == "tylko klasyczny błędny"
].head(3)

for _, row in transformer_wins.iterrows():
    text = str(row[TEXT_COL])[:250]
    true_label = "Obraźliwy" if row[LABEL_COL] == 1 else "Neutralny"
    print(f"\nTekst: {text}...")
    print(f"  Prawda: {true_label}")
    print("-" * 70)

print(f"\n\nPRZEWAGA KLASYCZNEGO – poprawny klasyczny, błędny Transformer")
print("=" * 70)

classic_wins = results_df[
    results_df["error_type"] == "tylko transformer błędny"
].head(3)

for _, row in classic_wins.iterrows():
    text = str(row[TEXT_COL])[:250]
    true_label = "Obraźliwy" if row[LABEL_COL] == 1 else "Neutralny"
    print(f"\nTekst: {text}...")
    print(f"  Prawda: {true_label}")
    print("-" * 70)

# Podsumowanie wyników
print("\n" + "=" * 60)
print("PODSUMOWANIE PROJEKTU")
print("=" * 60)
print(f"\nZadanie: Analiza sentymentu polskich recenzji produktów (Allegro)")
print(f"Zbiór danych: allegro-reviews (Hugging Face Datasets)")
print(f"Próbki testowe: {len(results_df)}")
print(f"\n{'Model':<30} {'Accuracy':>10} {'F1-score':>10}")
print("-" * 52)
print(f"{summary['classic_name']:<30} {summary['classic_accuracy']:>10.4f} {summary['classic_f1']:>10.4f}")
print(f"{'HerBERT (Transformer)':<30} {summary['transformer_accuracy']:>10.4f} {summary['transformer_f1']:>10.4f}")
print("-" * 52)

diff_acc = summary['transformer_accuracy'] - summary['classic_accuracy']
diff_f1 = summary['transformer_f1'] - summary['classic_f1']
print(f"{'Różnica (T - K)':<30} {diff_acc:>+10.4f} {diff_f1:>+10.4f}")

# Wnioski końcowe:

# Wyniki obydwu modeli znacznie się polepszyły po zastosowaniu wag.
# Zwykle takie działanie zwiększa liczbę False Positives, jednak w naszym przypadku,
# zgrubsza lepiej jest, gdy jest więcej fałszywych alarmów, niż gdy model przepuści więcej hejtu.
# Mimo wszystko, liczba False Positives zwiększyła się nieznacznie.

# Podejście klasyczne (TF-IDF + SVM/NB):
# - Szybki trening i inferencja.
# - Nie wymaga GPU.
# - Dobrze radzi sobie z tweetami zawierającymi wyraźne słowa kluczowe.
# - Problemy z ironią, sarkazmem, negacjami.

# Podejście Transformer (HerBERT):
# - Lepsze rozumienie kontekstu i niuansów językowych.
# - Wymaga GPU i więcej pamięci.
# - Dłuższy czas treningu.
# - Lepsze wyniki na trudnych przypadkach.

# Typowe błędy modeli:
# - Wpisy mieszane, np. zawierające elementy wulgarne, ale niebędące hejtem.
# - Ironia i sarkazm.
# - Krótkie, mało informacyjne tweety.
# - Wpisy z wieloma negacjami.
# - Tweety zawierające słowa spoza słownika, np. obelgi występujące tylko na stronach właśnie tego pokroju.
# - Używanie słów niewulgarnych w celach obraźliwych.

# Możliwe ulepszenia:
# - Więcej epok treningu dla Transformera.
# - Augmentacja danych.
# - Ensemble obu podejść.
# - Użycie większego modelu (np. Polish RoBERTa).
# - Zbalansowanie zbioru danych.
