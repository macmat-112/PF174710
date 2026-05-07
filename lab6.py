import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
import re

sns.set_theme(style="whitegrid")
dataset = load_dataset("allegro/klej-polemo2-in")
print(type(dataset))
print(dataset)
print("OUEK")

df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()
df_test = dataset["test"].to_pandas()

print(f"Zbiór treningowy:   {df_train.shape[0]} przykładów")
print(f"Zbiór walidacyjny:  {df_val.shape[0]} przykładów")
print(f"Zbiór testowy:      {df_test.shape[0]} przykładów")
print(f"\nŁącznie: {df_train.shape[0] + df_val.shape[0] + df_test.shape[0]} przykładów")

df_train.head(10)
     
# Mapowanie etykiet na czytelne nazwy
label_map = {
    "__label__meta_plus_m": "pozytywny",
    "__label__meta_minus_m": "negatywny",
    "__label__meta_zero": "neutralny",
    "__label__meta_amb": "ambiwalentny"
}

df_train["label"] = df_train["target"].map(label_map)

print("Rozkład klas w zbiorze treningowym:")
print(df_train["label"].value_counts())
print(f"\nProporcje:")
print(df_train["label"].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wykres słupkowy
colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
df_train["label"].value_counts().plot(kind="bar", ax=axes[0], color=colors, edgecolor="black")
axes[0].set_title("Rozkład klas -- zbiór treningowy", fontsize=13)
axes[0].set_xlabel("Sentyment")
axes[0].set_ylabel("Liczba przykładów")
axes[0].tick_params(axis="x", rotation=0)

# Wykres kołowy
df_train["label"].value_counts().plot(
    kind="pie", ax=axes[1], autopct="%1.1f%%", colors=colors, startangle=90
)
axes[1].set_ylabel("")
axes[1].set_title("Proporcje klas -- zbiór treningowy", fontsize=13)

plt.tight_layout()
plt.show()

df_train["num_chars"] = df_train["sentence"].str.len()
df_train["num_words"] = df_train["sentence"].str.split().str.len()

print("Statystyki długości (znaki):")
print(df_train["num_chars"].describe().round(1))
print(f"\nStatystyki długości (słowa):")
print(df_train["num_words"].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_train["num_chars"], bins=50, color="#3498db", edgecolor="black", alpha=0.7)
axes[0].axvline(df_train["num_chars"].median(), color="red", linestyle="--", label=f'mediana = {df_train["num_chars"].median():.0f}')
axes[0].set_title("Rozkład długości recenzji (znaki)", fontsize=13)
axes[0].set_xlabel("Liczba znaków")
axes[0].set_ylabel("Liczba recenzji")
axes[0].legend()

axes[1].hist(df_train["num_words"], bins=50, color="#2ecc71", edgecolor="black", alpha=0.7)
axes[1].axvline(df_train["num_words"].median(), color="red", linestyle="--", label=f'mediana = {df_train["num_words"].median():.0f}')
axes[1].set_title("Rozkład długości recenzji (słowa)", fontsize=13)
axes[1].set_xlabel("Liczba słów")
axes[1].set_ylabel("Liczba recenzji")
axes[1].legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(data=df_train, x="label", y="num_words", ax=ax, palette=colors,
            order=["pozytywny", "negatywny", "neutralny", "ambiwalentny"])
ax.set_title("Rozkład długości recenzji (słowa) w podziale na sentyment", fontsize=13)
ax.set_xlabel("Sentyment")
ax.set_ylabel("Liczba słów")
ax.set_ylim(0, df_train["num_words"].quantile(0.95))  # obcięcie ekstremalnych wartości

plt.tight_layout()
plt.show()

def tokenize_simple(text):
    """Prosta tokenizacja: lowercase + usunięcie znaków specjalnych."""
    text = text.lower()
    tokens = re.findall(r'\b[a-ząćęłńóśźż]+\b', text)
    return tokens

# Stopwords -- podstawowa lista polskich stopwordów
polish_stopwords = {
    "i", "w", "na", "z", "do", "nie", "się", "to", "jest", "że",
    "o", "jak", "ale", "co", "tak", "za", "po", "od", "już", "a",
    "przez", "by", "tym", "ze", "tego", "ten", "ta", "te", "bardzo",
    "też", "tylko", "czy", "był", "była", "było", "być", "są",
    "ma", "ich", "dla", "mnie", "mi", "ja", "sobie", "go", "pan",
    "przy", "u", "no", "jeszcze", "tu", "tam", "kiedy", "gdy",
    "wszystko", "może", "więc", "który", "która", "które", "których",
    "którzy", "mam", "będzie", "moim", "mojej", "mój", "moja",
    "bo", "ni", "lub"
}

all_words = []
for text in df_train["sentence"]:
    tokens = tokenize_simple(text)
    filtered = [t for t in tokens if t not in polish_stopwords and len(t) > 2]
    all_words.extend(filtered)

word_freq = Counter(all_words)
top_20 = word_freq.most_common(20)

print("20 najczęstszych słów (po usunięciu stopwordów):")
for word, count in top_20:
    print(f"  {word:20s} {count}")

words, counts = zip(*top_20)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(words)), counts, color="#9b59b6", edgecolor="black")
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xlabel("Częstość")
ax.set_title("20 najczęstszych słów w zbiorze treningowym", fontsize=13)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (label, ax) in enumerate(zip(
    ["pozytywny", "negatywny", "neutralny", "ambiwalentny"],
    axes.flatten()
)):
    subset = df_train[df_train["label"] == label]
    words_subset = []
    for text in subset["sentence"]:
        tokens = tokenize_simple(text)
        filtered = [t for t in tokens if t not in polish_stopwords and len(t) > 2]
        words_subset.extend(filtered)

    top_10 = Counter(words_subset).most_common(10)
    if top_10:
        w, c = zip(*top_10)
        ax.barh(range(len(w)), c, color=colors[idx], edgecolor="black")
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(w)
        ax.invert_yaxis()
    ax.set_title(f"Top 10 słów -- {label}", fontsize=12)
    ax.set_xlabel("Częstość")

plt.suptitle("Najczęstsze słowa w podziale na sentyment", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
