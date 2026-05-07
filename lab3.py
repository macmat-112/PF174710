# 0. Instalacja i import bibliotek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import random
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from nltk import ngrams
from wordcloud import WordCloud

# Polskie stop words — pobieramy z internetu
import urllib.request
url = "https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt"
response = urllib.request.urlopen(url)
stop_pl = set(response.read().decode("utf-8").splitlines())

print("\nWszystko gotowe!")
print(f"Załadowano {len(stop_pl)} polskich stop words.\n")

# 1. Ładowanie danych

# Załaduj wybrany plik CSV
df = pd.read_csv("recenzje_filmowe.csv")

# Wyświetl podstawowe informacje o zbiorze:
print(f"Rozmiar: {df.shape}\nKolumny: {df.columns}\nTypy danych:\n{df.dtypes}\nKilka pierwszych rzędów:")
print(f"{df.head()}\nKilka pierwszych tekstów:")
print(df["text"].head(3))

# 2. Tokenizacja i podstawowe statystyki korpusu

# Tokenizacja
df["tokens"] = df["text"].apply(lambda x: word_tokenize(x, language="polish"))
df["n_tokens"] = df["tokens"].apply(len)

# Wyświetl statystyki:
print(f"\nLiczba dokumentów: {df["text"].count()}")
print(f"Łączna liczba tokenów: {df["n_tokens"].sum()}")
unique = set()
for l in df["tokens"]:
    unique.update(l)
print(f"Liczba unikalnych tokenów: {len(unique)}")
print(f"Średnia długość dokumentu: {df["n_tokens"].mean()}")
print(f"Mediana długości: {df["n_tokens"].median()}")
print(f"Minimalna długość: {df["n_tokens"].min()}")
print(f"Maksymalna długość: {df["n_tokens"].max()}")

# 3. Rozkład długości dokumentów

plt.figure(figsize=(10, 5))
plt.hist(df["n_tokens"], bins=range(df["n_tokens"].min(), df["n_tokens"].max() + 2), edgecolor="black", alpha=0.7, align="left")
plt.xlabel("Liczba tokenów")
plt.ylabel("Liczba dokumentów")
plt.title("Rozkład długości dokumentów")
plt.axvline(df["n_tokens"].median(), color="red", linestyle="--", label=f"Mediana: {df['n_tokens'].median():.0f}")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Type–Token Ratio (TTR)

# Zbierz wszystkie tokeny
all_tokens = df["tokens"].explode().tolist()

# Filtrowanie i normalizacja
word_tokens = [t.lower() for t in all_tokens if t.isalpha() and len(t) > 1]

# Oblicz TTR
types = set(word_tokens)
ttr = len(types) / len(word_tokens)

# 5. Hapax legomena

freq = Counter(word_tokens)
hapax = [w for w, c in freq.items() if c == 1]
hap_len = len(hapax)
print(f"\nIlość hapax legomena: {hap_len}, skład procentowy: {hap_len * 100 / len(unique)}, przykładowe trzy: {random.sample(hapax, 3)}")

# 6. Najczęstsze słowa

top = Counter(word_tokens).most_common(20)
word_tokens_nostop = [t for t in word_tokens if t not in stop_pl]
top_without = Counter(word_tokens_nostop).most_common(20)

print(f"\nNajczęstsze 20 słów:\n{'-' * 35}-+-{'-' * 35}")
print(f"{"Ze stop words":<35} | Bez stop words\n{'-' * 35}-+-{'-' * 35}")
for i in range(len(top)):
    print(f"{top[i][0]:<15}{'#' * top[i][1]:<20} | {top_without[i][0]:<15}{'#' * top_without[i][1]:<20}")

# 7. Statystyki na poziomie dokumentu

df["ttr"] = df["tokens"].apply(lambda t: len(set(t)) / len(t) if len(t) > 0 else 0)
df["avg_word_len"] = df["tokens"].apply(lambda l: sum(len(t) for t in l) / len(l) if len(l) > 0 else 0)
df["stop_ratio"] = df["tokens"].apply(lambda l: sum(1 for t in l if t in stop_pl) * 100 / len(l) if len(l) > 0 else 0)

print(f"\nStatystyki na poziomie dokumentu:\n{df[["n_tokens", "ttr", "avg_word_len", "stop_ratio"]].describe()}")

# 8. Wizualizacja: chmury słów

# Chmura słów dla całego korpusu
text_all = " ".join(word_tokens)
wc = WordCloud(width=800, height=400,
               background_color="white",
               max_words=150,
               colormap="viridis",
               stopwords=stop_pl)
wc.generate(text_all)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Chmura słów")
plt.show()

all_tokens_pos = df[df["label"] == "pozytywna"]["tokens"].explode().tolist()
all_tokens_neg = df[df["label"] == "negatywna"]["tokens"].explode().tolist()
word_tokens_pos = [t.lower() for t in all_tokens_pos if t.isalpha() and len(t) > 1]
word_tokens_neg = [t.lower() for t in all_tokens_neg if t.isalpha() and len(t) > 1]
text_pos = " ".join(word_tokens_pos)
text_neg = " ".join(word_tokens_neg)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

wc.generate(text_pos)
axes[0].imshow(wc, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Opinie Pozytywne", fontsize=16)

wc.generate(text_neg)
axes[1].imshow(wc, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Opinie Negatywne", fontsize=16)

plt.tight_layout()
plt.show()

# 9. Wizualizacja: histogramy częstości

top_words = Counter(word_tokens_nostop).most_common(25)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.barh(range(len(words)), counts, color="steelblue")
plt.yticks(range(len(words)), words)
plt.xlabel("Częstość")
plt.title("25 najczęstszych słów (bez stop words)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 10. N-gramy

# Bigramy — wszystkie
bigrams = list(ngrams(word_tokens, 2))
bigram_freq = Counter(bigrams)
print("\nBigramy:")
for b in bigram_freq.most_common(15):
    print(b)

# Bigramy — bez stop words
bigrams_filtered = list(ngrams(word_tokens_nostop, 2))
bigram_freq_f = Counter(bigrams_filtered)
print("\nBigramy (bez stop words):")
for b in bigram_freq_f.most_common(15):
    print(b)

# (Opcjonalnie) Trigramy
trigrams = list(ngrams(word_tokens, 3))
trigram_freq = Counter(bigrams)
print("\nTrigramy:")
for t in trigram_freq.most_common(15):
    print(t)

trigrams_filtered = list(ngrams(word_tokens_nostop, 3))
trigram_freq_f = Counter(trigrams_filtered)
print("\nTrigramy (bez stop words):")
for t in trigram_freq_f.most_common(15):
    print(t)

# 11. Prawo Zipfa

# Wykres log-log prawa Zipfa
ranks = range(1, len(freq) + 1)
frequencies = sorted(freq.values(), reverse=True)

plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, linewidth=0.8)
plt.xlabel("Ranga (log)")
plt.ylabel("Częstość (log)")
plt.title("Prawo Zipfa — wykres log-log")
plt.grid(True, alpha=0.3)

# Linia teoretyczna
zipf_theoretical = [frequencies[0] / r for r in ranks]

plt.loglog(ranks, zipf_theoretical, "--", color="red", label=r"Zipf teoretyczny ($\alpha = 1$)", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Pokrycie top-N słów
top100 = freq.most_common(100)
coverage = sum(c for _, c in top100) / sum(freq.values())
print(f"\nTop 100 słów pokrywa {coverage*100:.1f}% wszystkich tokenów")

top1000 = freq.most_common(1000)
coverage = sum(c for _, c in top1000) / sum(freq.values())
print(f"Top 1000 słów pokrywa {coverage*100:.1f}% wszystkich tokenów\n")

# 12a. Porównanie zbiorów

df2 = pd.read_csv("recenzje_restauracji.csv")
m1 = sum(len(t) for t in df["text"]) / df["text"].count()
m2 = sum(len(t) for t in df2["text"]) / df2["text"].count()
df2["tokens"] = df2["text"].apply(lambda x: word_tokenize(x, language="polish"))
all_tokens_2 = df2["tokens"].explode().tolist()
word_tokens_2 = [t.lower() for t in all_tokens_2 if t.isalpha() and len(t) > 1]
types_2 = set(word_tokens_2)
ttr_2 = len(types_2) / len(word_tokens_2)
freq_2 = Counter(word_tokens_2)
hapax_2 = [w for w, c in freq_2.items() if c == 1]
hap_len_2 = len(hapax_2)
word_tokens_nostop_2 = [t for t in word_tokens_2 if t not in stop_pl]
top10_without = Counter(word_tokens_nostop).most_common(10)
top10_without_2 = Counter(word_tokens_nostop_2).most_common(10)

print(f"{"Recenzje:":<26} | {"Filmowe":<15} | Restauracji")
print(f"{'-' * 26}-+-{'-' * 15}-+-{'-' * 15}")
print(f"{"Średnia długość dokumentu:":<26} | {m1:<15} | {m2}")
print(f"{"TTR korpusu:":<26} | {ttr:.5f}{' ' * 8} | {ttr_2:.5f}")
print(f"{"Liczba hapax legomena":<26} | {hap_len:<15} | {hap_len_2}")
print(f"\nNajczęstsze 10 słów (bez stop words):\n{'-' * 35}-+-{'-' * 35}")
print(f"{"Recenzje filmowe":<35} | Recenzje restauracji\n{'-' * 35}-+-{'-' * 35}")
for i in range(len(top10_without)):
    print(f"{top10_without[i][0]:<15}{'#' * top10_without[i][1]:<20} | {top10_without_2[i][0]:<15}{'#' * top10_without_2[i][1]:<20}")

# 12b. Analiza sentymentu — różnice słownikowe

pos_20 = Counter([t for t in word_tokens_pos if t not in stop_pl]).most_common(20)
neg_20 = Counter([t for t in word_tokens_neg if t not in stop_pl]).most_common(20)
set_pos = set()
set_neg = set()
for i in range(20):
    set_pos.add(pos_20[i][0])
    set_neg.add(neg_20[i][0])
s = ""
for t in set_pos - set_neg:
    s += f"{t}, "
print(f"\nSłowa z top 20 słów z recenzji pozytywnych występujące tylko w nich: {s[:-2]}.")
s = ""
for t in set_neg - set_pos:
    s += f"{t}, "
print(f"\nSłowa z top 20 słów z recenzji negatywnych występujące tylko w nich: {s[:-2]}.")

# 12c. Strefy Zipfa (Luhn)

all = set(word_tokens)
no_stop = set(word_tokens_nostop)
stop = all - no_stop
hap = set(hapax)
no_stop_no_hap = no_stop - hap
# print(f"\nGórna strefa: {stop}")
# print(f"\nDolna strefa: {hap}")
print(f"\nSłowa ze strefy środkowej: {no_stop_no_hap}")
# print(f"\nWSZYSTKIE słowa: {all}")
