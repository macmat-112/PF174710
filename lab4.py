import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Zadania A

recenzje_ksiazek = [
    # FANTASY
    "Magiczny swiat pelen elfow i smokow, wciagajaca fabula od pierwszej strony",
    "Bohater wyrusza w epicka podroz by pokonac mrocznego wladce",
    "Swietny system magii i budowanie swiata, autor ma niesamowita wyobraznie",
    "Mroczne fantasy z moralnymi dylematami, postacie niejednoznaczne i ciekawe",
    "Piekne opisy krain i stworzen, przypomina najlepsze dziela Tolkiena",
    "Zaklecia i magiczne artefakty tworza fascynujacy element fabuly",

    # KRYMINAL
    "Detektyw prowadzi sledztwo w sprawie zagadkowego morderstwa w zamknietym pokoju",
    "Trzymajacy w napieciu thriller z zaskakujacym zwrotem akcji na koncu",
    "Policjant tropi seryjnego morderce po sladach zostawionych na miejscu zbrodni",
    "Zimny kryminal skandynawski pelen mrocznych tajemnic i zlozonej intrygi",
    "Sledztwo w malym miasteczku odslania mroczne sekrety mieszkancow",
    "Genialny detektyw rozwiazuje sprawe pozornie idealnej zbrodni",
]

kategorie_ksiazek = ["Fantasy"]*6 + ["Kryminal"]*6

print(f"\nKorpus: {len(recenzje_ksiazek)} dokumentow")
for i, r in enumerate(recenzje_ksiazek):
    print(f"  D{i:2d} [{kategorie_ksiazek[i]:>9s}]: {r}")

count_vec = CountVectorizer()
X_bow = count_vec.fit_transform(recenzje_ksiazek)
slownik = count_vec.get_feature_names_out()

print(f"\nRozmiar macierzy BoW: {X_bow.shape}  (dokumenty x termy)")
print(f"Rozmiar slownika: {len(slownik)} unikalnych termow")
print(f"Niezerowych wpisow: {X_bow.nnz} / {X_bow.shape[0]*X_bow.shape[1]}  "
      f"({X_bow.nnz / (X_bow.shape[0]*X_bow.shape[1]):.1%} gestosci)")

tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(recenzje_ksiazek)

df_tfidf = pd.DataFrame(
    X_tfidf.toarray().round(3),
    columns=tfidf_vec.get_feature_names_out(),
    index=[f"D{i}" for i in range(len(recenzje_ksiazek))]
)
print("\nMacierz TF-IDF:")
print(df_tfidf.T)

N = 3
print(f"\nTOP-{N} NAJWAZNIEJSZE CECHY W KAZDYM DOKUMENCIE (wg TF-IDF)")
print("=" * 70)

for i, og in enumerate(recenzje_ksiazek):
    wiersz = X_tfidf[i].toarray().flatten()
    top_idx = wiersz.argsort()[::-1][:N]
    cechy = [(slownik[j], round(wiersz[j], 3)) for j in top_idx]
    print(f"\nD{i} [{kategorie_ksiazek[i]:>11s}]: {og[:55]}...")
    for nazwa, waga in cechy:
        print(f"       {waga:.3f}  <<{nazwa}>>")

freq_dok = np.asarray((X_bow > 0).sum(axis=0)).flatten()
freq_total = np.asarray(X_bow.sum(axis=0)).flatten()
srednia_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()

df_stats = pd.DataFrame({
    "term":          slownik,
    "wystapienia":   freq_total,
    "w_ilu_dok":     freq_dok,
    "sr_tfidf":      srednia_tfidf.round(4),
}).sort_values("sr_tfidf", ascending=False).reset_index(drop=True)

print("\nRANKING TERMOW (wg sredniego TF-IDF):")
print(df_stats.to_string(index=False))

cos_sim = cosine_similarity(X_tfidf)

df_sim = pd.DataFrame(
    cos_sim.round(3),
    columns=[f"D{i}" for i in range(len(recenzje_ksiazek))],
    index=[f"D{i}" for i in range(len(recenzje_ksiazek))]
)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cos_sim, cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(recenzje_ksiazek)))
ax.set_yticks(range(len(recenzje_ksiazek)))
ax.set_xticklabels([f"D{i}" for i in range(len(recenzje_ksiazek))])
ax.set_yticklabels([f"D{i}\n{k}" for i, k in enumerate(kategorie_ksiazek)])
plt.colorbar(im, label="Cosine similarity")
plt.title("Podobienstwo cosinusowe dokumentow (TF-IDF)")

for i in range(len(recenzje_ksiazek)):
    for j in range(len(recenzje_ksiazek)):
        ax.text(j, i, f"{cos_sim[i,j]:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.show()

konfig = [
    {"ngram_range": (1,1), "max_features": None, "min_df": 1},
    {"ngram_range": (1,2), "max_features": None, "min_df": 1},
    {"ngram_range": (1,2), "max_features": 20,   "min_df": 1},
    {"ngram_range": (1,2), "max_features": None, "min_df": 2},
    {"ngram_range": (1,3), "max_features": None, "min_df": 1},
]

print(f"\n{'n-gram':>10} | {'max_feat':>10} | {'min_df':>6} | {'Cech':>8} | {'Rzadkosc':>10}")
print("-" * 60)

for cfg in konfig:
    vec = TfidfVectorizer(**cfg)
    X = vec.fit_transform(recenzje_ksiazek)
    sparsity = 1 - X.nnz / (X.shape[0] * X.shape[1])
    print(f"{str(cfg['ngram_range']):>10} | {str(cfg['max_features']):>10} | "
          f"{cfg['min_df']:>6} | {X.shape[1]:>8} | {sparsity:>10.4f}")

# Zadania B

posty = [
    # SPORT (0-4)
    "Trening interwalowy to najlepszy sposob na poprawe kondycji biegowej",
    "Reprezentacja zdobyla zloty medal na mistrzostwach swiata w siatkowce",
    "Nowy rekord swiata w maratonie pobity o trzy sekundy",
    "Trener oglosil powolania na zgrupowanie kadry przed meczem",
    "Dieta i regeneracja sa rownie wazne jak sam trening sportowy",

    # TECHNOLOGIA (5-9)
    "Premiera nowego smartfona z aparatem o rozdzielczosci stu megapikseli",
    "Sztuczna inteligencja zmienia sposob w jaki pracujemy i uczymy sie",
    "Aktualizacja systemu operacyjnego przynosi nowe funkcje bezpieczenstwa",
    "Robot wykorzystujacy sztuczna inteligencje pomaga w diagnostyce medycznej",
    "Nowy laptop z procesorem najnowszej generacji i ekranem OLED",

    # KUCHNIA (10-14)
    "Domowy chleb na zakwasie wymaga cierpliwosci ale smakuje wysmienicie",
    "Przepis na szarlotke z kruszonka i cynamonem na jesienne wieczory",
    "Kuchnia azjatycka laczy ostre przyprawy z delikatnymi sosami",
    "Pieczony kurczak z ziolami prowansalskimi i pieczonymi warzywami",
    "Sezon na grzyby to idealny czas na domowy krem z borowikow",
]

kategorie_postow = ["Sport"]*5 + ["Tech"]*5 + ["Kuchnia"]*5

print(f"\nKorpus: {len(posty)} dokumentow")
for i, p in enumerate(posty):
    print(f"  D{i:2d} [{kategorie_postow[i]:>7s}]: {p}")

p_tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
p_X_tfidf = p_tfidf_vec.fit_transform(posty)
p_slownik = p_tfidf_vec.get_feature_names_out()

p2_tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
p2_X_tfidf = p2_tfidf_vec.fit_transform(posty)
p2_slownik = p2_tfidf_vec.get_feature_names_out()

p2_df_tfidf = pd.DataFrame(
    p2_X_tfidf.toarray().round(3),
    columns=p2_tfidf_vec.get_feature_names_out(),
    index=[f"D{i}" for i in range(len(posty))]
)
print("\nMacierz TF-IDF z bigramami:")
print(p2_df_tfidf.T)

print(f"\nWymiary TF-IDF z bigramami: {p2_X_tfidf.shape}")
print(f"Wymiary TF-IDF z unigramami: {p_X_tfidf.shape}")

N = 5
print(f"\nTOP-{N} NAJWAZNIEJSZYCH CECH W KAZDYM DOKUMENCIE (wg TF-IDF) - UNIGRAMY")
print("=" * 70)

for i, og in enumerate(posty):
    wiersz = p_X_tfidf[i].toarray().flatten()
    top_idx = wiersz.argsort()[::-1][:N]
    cechy = [(p_slownik[j], round(wiersz[j], 3)) for j in top_idx]
    print(f"\nD{i} [{kategorie_postow[i]:>11s}]: {og[:55]}...")
    for nazwa, waga in cechy:
        print(f"       {waga:.3f}  <<{nazwa}>>")

print(f"\nTOP-{N} NAJWAZNIEJSZYCH CECH W KAZDYM DOKUMENCIE (wg TF-IDF) - BIGRAMY")
print("=" * 70)

for i, og in enumerate(posty):
    wiersz = p2_X_tfidf[i].toarray().flatten()
    top_idx = wiersz.argsort()[::-1][:N]
    cechy = [(p2_slownik[j], round(wiersz[j], 3)) for j in top_idx]
    print(f"\nD{i} [{kategorie_postow[i]:>11s}]: {og[:55]}...")
    for nazwa, waga in cechy:
        print(f"       {waga:.3f}  <<{nazwa}>>")

def znajdz_podobne(query, n=3):
    q_vec = p2_tfidf_vec.transform([query])
    sim = cosine_similarity(q_vec, p2_X_tfidf).flatten()
    top_idx = sim.argsort()[::-1][:n]
    return [posty[i] for i in top_idx]

zapytania = ["trening silowy i odzywanie sportowcow", "nowy smartfon z aparatem i sztuczna inteligencja", "przepis na domowe ciasto drozdzowe"]
for i in zapytania:
    print(f"\nDla zapytania \"{i}\":")
    for j in znajdz_podobne(i):
        print(j)
