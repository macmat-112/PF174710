# 1. Dane i przetwarzanie wstępne

# Tytuł: Analiza sentymentu polskich wpisów w celu wykrycia mowy nienawiści.
# Autor: Maciej Matyjasek
# Opis: Projekt realizuje analizę sentymentu polskich wpisów na Twitterze/X w celu wykrycia tzw. hate speech.
#       Porównujemy podejście klasyczne (TF-IDF + SVM) z podejściem opartym na Transformerach (HerBERT).
# Źródło danych: Zbiór Allegro KLEJ-CBD z Hugging Face Datasets, bazujący na zbiorze PolEval 2019 - publicznie 
#                dostępny zbiór wpisów z Twittera/X z ocenami: 0 (non-harmful), 1 (harmful).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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


def clean_text(text):
    """Czyszczenie śmieciowych wartości z tekstu: wzmianek (@user), URL-i, tagów HTML."""
    if not isinstance(text, str):
        return ""
    
    # Usunięcie wzmianek
    text = re.sub(r"@anonymized_account ", "", text)
    
    # Usunięcie URL-i
    text = re.sub(r"http\S+|www\S+", "", text)

    # Usunięcie tagów HTML
    text = re.sub(r"<.*?>", "", text)

    return text


def preprocess_text(text):
    """Przetwarzanie wstępne tekstu: lowercase, lematyzacja, usunięcie znaków specjalnych i stopwords."""
    # Zamiana na małe litery
    text = text.lower()

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
print(f"Po przetworzeniu: {preprocess_text(clean_text(sample_text))[:200]}")

# Stworzenie zbioru walidacyjnego
df_train, df_val = train_test_split(
    df_train,
    test_size=1000,
    random_state=42,
    stratify=df_train['target']
)

# Czyszczenie tekstu
print("Czyszczenie tekstu...")
df_train["text_clean"] = df_train["sentence"].astype(str).apply(clean_text)
df_val["text_clean"] = df_val["sentence"].astype(str).apply(clean_text)
df_test["text_clean"] = df_test["sentence"].astype(str).apply(clean_text)

# Usunięcie pustych rekordów po czyszczeniu
df_train = df_train[df_train["text_clean"].str.len() > 0].reset_index(drop=True)
df_val = df_val[df_val["text_clean"].str.len() > 0].reset_index(drop=True)
df_test = df_test[df_test["text_clean"].str.len() > 0].reset_index(drop=True)

# Przetwarzanie wstępne
print("Przetwarzanie tekstu...")
df_train["text_processed"] = df_train["text_clean"].apply(preprocess_text)
df_val["text_processed"] = df_val["text_clean"].apply(preprocess_text)
df_test["text_processed"] = df_test["text_clean"].apply(preprocess_text)

print(f"Rekordów po przetworzeniu: {len(df_train) + len(df_val) + len(df_test)}")

print(f"\nRozmiary zbiorów:")
print(f"  Treningowy: {len(df_train)}")
print(f"  Walidacyjny: {len(df_val)}")
print(f"  Testowy: {len(df_test)}")

print(f"\nRozkład ocen wpisów w zbiorze treningowym:")
print(df_train["target"].value_counts(normalize=True))

# Zapis przetworzonych danych do plików CSV
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

print("\nDane zapisane do plików: train.csv, test.csv")
print("Kolumny w pliku:")
print(f"  'sentence' - oryginalny tekst")
print(f"  'text_clean' - wyczyszczony tekst (dla podejścia opartego na Transformerach)")
print(f"  'text_processed' - przetworzony tekst (dla podejścia klasycznego)")
print(f"  'target' - etykieta (0=neutralny, 1=obraźliwy)")
