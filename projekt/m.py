# 1. Dane i przetwarzanie wstępne

# Tytuł: Analiza sentymentu polskich wpisów w celu wykrycia mowy nienawiści.
# Autor: Maciej Matyjasek
# Opis: Projekt realizuje analizę sentymentu polskich wpisów na Twitterze/X w celu wykrycia tzw. hate speech.
#       Porównujemy podejście klasyczne (TF-IDF + SVM) z podejściem opartym na Transformerach (HerBERT).
# Źródło danych: Zbiór Allegro KLEJ-CBD z Hugging Face Datasets, bazujący na zbiorze PolEval 2019 - publicznie 
#                dostępny zbiór wpisów z Twittera/X z ocenami: 0 (non-harmful), 1 (harmful).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from datasets import load_dataset

# Ustawienia wyświetlania
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option("display.max_colwidth", 200)

# Wczytanie zbioru CBD z benchmarku KLEJ (który bazuje na PolEval 2019)
dataset = load_dataset("allegro/klej-cbd")

# Podgląd struktury
print(f"\nStruktura zbioru danych:\n{dataset}")

# Konwersja do DataFrame
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()
print(f"\nInformacje o zbiorach po konwersji do DataFrame:")
print(f"{"":<30}|{"train":<30}|{"test":<30}")
print(f"{"-" * 30}+{"-" * 30}+{"-" * 30}")
print(f"{"Liczba rekordów":<30}|{len(df_train):<30}|{len(df_test):<30}")
print(f"{"Kolumny":<30}|{str(list(df_train.columns)):<30}|{str(list(df_test.columns)):<30}")

# Do EDA użyjemy treningowego zbioru danych i kolumn: tekst wpisu ("sentence") i ocena ("target")
# Wyświetlenie kolumn i pierwszych wartości po konwersji do DataFrame
print(f"\nPodgląd kolumn i typów ze zbioru treningowego:\n{df_train.dtypes}")
print(f"\nPierwsze wartości ze zbioru treningowego:\n{df_train.head(10)}")

# Czyszczenie wartości
# df_train["sentence"] = df_train["sentence"].astype(str).apply(lambda x: x.replace("@anonymized_account", ''))
# print(f"\nPierwsze wartości ze zbioru treningowego (po wyczyszczeniu):\n{df_train.head(10)}")

# Wyświetlenie rozkładu ocen wpisów: 0 - neutralny, 1 - obraźliwy
print(f"\nRozkład ocen:\n{df_train["target"].value_counts()}")

# Rozkład ocen wpisów
plt.figure(figsize=(10, 8))
plt.bar(sorted(df_train["target"].unique()), df_train["target"].value_counts(), color=["mediumseagreen", "salmon"], label=["Neutralny", "Obraźliwy"])
plt.xlabel("Oceny wpisów")
plt.ylabel("Liczba wpisów")
plt.xticks(sorted(df_train["target"].unique()), ["Neutralny", "Obraźliwy"], rotation=0)
plt.title("Binarny rozkład ocen wpisów")
plt.legend()
plt.tight_layout()
plt.show()

# Długość wpisów
df_train["text_length"] = df_train["sentence"].astype(str).apply(len)
df_train["word_count"] = df_train["sentence"].astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for target, color, label in [(0, "mediumseagreen", "Neutralny"), (1, "salmon", "Obraźliwy")]:
    subset = df_train[df_train["target"] == target]
    axes[0].hist(subset["word_count"], bins=len(subset["word_count"].unique()), align="left", alpha=0.6, label=label, color=color)
    axes[1].hist(subset["text_length"], bins=len(subset["text_length"].unique()), align="left", alpha=0.6, label=label, color=color)

axes[0].set_title("Rozkład liczby słów")
axes[0].set_xlabel("Liczba słów")
axes[0].legend()

axes[1].set_title("Rozkład długości tekstu (znaki)")
axes[1].set_xlabel("Liczba znaków")
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nStatystyki długości recenzji (słowa):")
print(df_train.groupby("target")["word_count"].describe())

# Ładowanie modelu spaCy dla języka polskiego
nlp = spacy.load("pl_core_news_sm", disable=["ner", "parser"])


def preprocess_text(text):
    """Przetwarzanie wstępne tekstu: czyszczenie, lematyzacja, usunięcie stopwords."""
    if not isinstance(text, str):
        return ""

    # Zamiana na małe litery
    text = text.lower()

    # Usunięcie URL-i
    text = re.sub(r"http\S+|www\S+", "", text)

    # Usunięcie tagów HTML
    text = re.sub(r"<.*?>", "", text)

    # Usunięcie znaków specjalnych (zostawiamy litery i spacje)
    text = re.sub(r"[^a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s]", "", text)

    # Lematyzacja i usunięcie stopwords za pomocą spaCy
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 1
    ]

    return " ".join(tokens)


# Przykład działania
sample_text = df_train["sentence"].iloc[0]
print(f"\nOryginał: {sample_text[:200]}")
print(f"Po przetworzeniu: {preprocess_text(sample_text)[:200]}")

# Przetwarzanie wstępne dla podejścia klasycznego
# (Transformer będzie korzystał z surowego tekstu)
print("Przetwarzanie tekstu... (może potrwać kilka minut)")
df_train["text_clean"] = df_train["sentence"].astype(str).apply(preprocess_text)
df_test["text_clean"] = df_test["sentence"].astype(str).apply(preprocess_text)

# Usunięcie pustych rekordów po przetworzeniu
df_train = df_train[df_train["text_clean"].str.len() > 0].reset_index(drop=True)
df_test = df_test[df_test["text_clean"].str.len() > 0].reset_index(drop=True)
print(f"Rekordów po przetworzeniu: {len(df_train) + len(df_test)}")

print(f"\nRozmiary zbiorów:")
print(f"  Treningowy: {len(df_train)}")
print(f"  Testowy: {len(df_test)}")

print(f"\nRozkład ocen wpisów w zbiorze treningowym:")
print(df_train["target"].value_counts(normalize=True))

# Zapis przetworzonych danych do plików CSV
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)

print("\nDane zapisane do plików: train.csv, test.csv")
print("Kolumny w pliku:")
print(f"  'sentence' - oryginalny tekst (dla Transformerów)")
print(f"  'text_clean' - przetworzony tekst (dla podejścia klasycznego)")
print(f"  'target' - etykieta (0=negatywny, 1=pozytywny)")
