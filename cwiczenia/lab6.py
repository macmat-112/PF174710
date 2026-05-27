import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
import re

sns.set_theme(style="whitegrid")

label_map = {
    "__label__meta_plus_m": "pozytywny",
    "__label__meta_minus_m": "negatywny",
    "__label__meta_zero": "neutralny",
    "__label__meta_amb": "ambiwalentny"
}

# Zadania cz. A

dataset_in = load_dataset("allegro/klej-polemo2-in")
df_in = dataset_in["train"].to_pandas()
df_in["label"] = df_in["target"].map(label_map)

# Krok 1: Załaduj dataset Out-of-Domain
dataset_out = load_dataset("allegro/klej-polemo2-out")
df_out = dataset_out["train"].to_pandas()
df_out["label"] = df_out["target"].map(label_map)

# Krok 2: Podstawowe statystyki
print(df_out.shape, df_out["target"].nunique(), df_out["target"].value_counts())

# Krok 3: Porównanie rozkładów klas In-Domain vs Out-of-Domain
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
df_out["label"].value_counts().plot(kind="bar", ax=axes[0], color=colors, edgecolor="black")
axes[0].set_title("Rozkład klas -- Out-of-Domain", fontsize=13)
axes[0].set_xlabel("Sentyment")
axes[0].set_ylabel("Liczba przykładów")
axes[0].tick_params(axis="x", rotation=0)

df_in["label"].value_counts().plot(kind="bar", ax=axes[1], color=colors, edgecolor="black")
axes[1].set_title("Rozkład klas -- In-Domain", fontsize=13)
axes[1].set_xlabel("Sentyment")
axes[1].set_ylabel("Liczba przykładów")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# Krok 4: Średnia i mediana długości w podziale na klasy
df_out["num_words"] = df_out["sentence"].str.split().str.len()
print(df_out.groupby("label")["num_words"].agg(["mean", "median"]))

# Krok 5: Histogram porównawczy In-Domain vs Out-of-Domain
df_in["num_words"] = df_in["sentence"].str.split().str.len()
plt.figure(figsize=(10, 5))
plt.hist([df_out["num_words"], df_in["num_words"]], label=["a", "b"], alpha=0.7)
plt.xlabel("Liczba słów")
plt.ylabel("Liczba recenzji")
plt.title("Rozkład długości recenzji (słowa)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()

# Krok 6: Najdłuższa i najkrótsza recenzja
a, b = df_out["num_words"].idxmax(), df_out["num_words"].idxmin()
print(f"\nNajdłuższa recenzja: {df_out["sentence"][a]}\nNajkrótsza recenzja: {df_out["sentence"][b]}")

# Zadania cz. B

# Krok 1: Załaduj Allegro Reviews
# dataset_ar = load_dataset("allegro/klej-allegro-reviews")
# df_ar = dataset_ar["train"].to_pandas()
# print(dataset_ar)
# df_ar.head()
