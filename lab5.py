import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import spacy
from datasets import load_dataset

# Zadania cz. A
model = api.load("glove-wiki-gigaword-100")

# Krok 1: Wektor slowa "university"
vec = model['university']
# Wyswietl: ksztalt i pierwsze 10 wartosci
print(f"Ksztalt: {vec.shape}")
print(f"Pierwsze 10 wartosci: {vec[:10]}\n")

# Krok 2: 10 najblizszych sasiadow dla "science", "music", "football"
for w in ["science", "music", "football"]:
    similar = model.most_similar(w, topn=10)
    print(f"Najbardziej podobne do \"{w}\":")
    for word, score in similar:
        print(f"  {word:15s} {score:.4f}")
    print()
# wyniki mają sens: dla "science" zwróciło słowa z kategorii naukowych, dla "music" z kategorii muzycznych, a dla "football" z kategorii sportowych.

# Krok 3: Podobienstwo cosinusowe miedzy parami
for p in [("doctor", "nurse"), ("doctor", "airplane"), ("happy", "sad"), ("happy", "joyful")]:
    sim = model.similarity(p[0], p[1])
    print(f"Podobienstwo {p[0]}-{p[1]}: {sim:.4f}")
# Najbliższe są zawody medyczne ("doctor", "nurse"), potem emocje ("happy", "sad"), ("happy", "joyful"), mało podobne są ("doctor", "airplane").

# Krok 4: doesnt_match
print(f"\nDo [\"apple\", \"banana\", \"orange\", \"car\"] nie pasuje: {model.doesnt_match(["apple", "banana", "orange", "car"])}")
print(f"Do [\"dog\", \"cat\", \"mouse\", \"table\"] nie pasuje: {model.doesnt_match(["dog", "cat", "mouse", "table"])}\n")

# Krok 5: Analogie wektorowe
# japan - tokyo + paris = ?
# teacher - school + hospital = ?
# slow - slower + faster = ?
for p, n in [(["japan", "paris"], ["tokyo"]), (["teacher", "hospital"], ["school"]), (["slow", "faster"], ["slower"])]:
    result = model.most_similar(positive=p, negative=n, topn=1)[0]
    print(f"{p[0]} - {n[0]} + {p[1]} = {result[0]} (score: {result[1]:.4f})")

# Krok 6: Wizualizacja t-SNE dla 4 grup slow
# Grupy: sporty (np. football, tennis, basketball, swimming, volleyball)
#        zawody (np. doctor, teacher, engineer, lawyer, nurse)
#        jedzenie (np. pizza, pasta, bread, cheese, rice)
#        emocje (np. happy, sad, angry, scared, surprised)
groups = {
    "sports": ["football", "tennis", "basketball", "swimming", "volleyball"],
    "professions": ["doctor", "teacher", "engineer", "lawyer", "nurse"],
    "food": ["pizza", "pasta", "bread", "cheese", "rice"],
    "emotions": ["happy", "sad", "angry", "scared", "surprised"]
}

words, vectors, labels = [], [], []
for label, group in groups.items():
    for w in group:
        words.append(w)
        vectors.append(model[w])
        labels.append(label)

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(np.array(vectors))

fig, ax = plt.subplots(figsize=(10, 8))
colors_map = {"sports": "red", "professions": "blue", "food": "green", "emotions": "yellow"}
plotted_labels = set()

for i, w in enumerate(words):
    group = labels[i]
    ax.scatter(coords[i, 0], coords[i, 1],
               c=colors_map[group], s=60,
               label=group if group not in plotted_labels else "")
    plotted_labels.add(group)
    ax.annotate(w, (coords[i, 0]+1, coords[i, 1]+1), fontsize=9)

ax.legend()
ax.set_title("Embeddingi GloVe -- wizualizacja t-SNE")
plt.tight_layout()
plt.show()

# Zadania cz. B
nlp_pl = spacy.load("pl_core_news_md")

zdania = [
    # FINANSE (0-3)
    "Kurs euro wzrosl do najwyzszego poziomu od poczatku roku",
    "Gielda warszawska zamknela sesje na plusie po dobrych wynikach spolek",
    "Inflacja spadla do najnizszego poziomu od dwoch lat",
    "Bank centralny podjal decyzje o obnizeniu stop procentowych",

    # TECHNOLOGIA (4-7)
    "Premiera nowego systemu operacyjnego przyciagnela miliony uzytkownikow",
    "Sztuczna inteligencja zmienia sposob tworzenia oprogramowania",
    "Firma technologiczna zaprezentowala innowacyjny chip do smartfonow",
    "Cyberbezpieczenstwo staje sie priorytetem dla polskich przedsiebiorstw",

    # TURYSTYKA (8-11)
    "Sezon turystyczny w Zakopanem zapowiada sie rekordowo",
    "Nowe polaczenia lotnicze z Polski do Azji Poludniowo-Wschodniej",
    "Krakow znalazl sie w czolowce najpopularniejszych miast do odwiedzenia",
    "Polskie wybrzeze przyciaga coraz wiecej turystow z zagranicy",
]

kategorie_zdan = ["Finanse"]*4 + ["Technologia"]*4 + ["Turystyka"]*4

print(f"\nKorpus: {len(zdania)} zdan")
for i, z in enumerate(zdania):
    print(f"  D{i:2d} [{kategorie_zdan[i]:>12s}]: {z}")

# Krok 1: Macierz podobienstwa miedzy zdaniami
cos_sim = cosine_similarity([nlp_pl(t).vector for t in zdania])
print(f"\nMacierz podobieństwa:\n{cos_sim.round(3)}\n")

# Krok 2: Dla kazdego zdania -- najblizsze i najdalsze
cos_sim_max = cos_sim.copy()
np.fill_diagonal(cos_sim_max, -np.inf)
max_i = np.argmax(cos_sim_max, axis=1)
min_i = np.argmin(cos_sim, axis=1)

for i in range(len(cos_sim)):
    max_v = cos_sim[i, max_i[i]]
    min_v = cos_sim[i, min_i[i]]
    print(f"Zdanie D{i}:")
    print(f"Najbliższe: D{max_i[i]} (wynik: {max_v:.3f})")
    print(f"Najdalsze: D{min_i[i]} (wynik: {min_v:.3f})\n")

# Krok 3: Funkcja znajdz_najblizsze(query, zdania, nlp, n=3)
def znajdz_najblizsze(query, zdania, nlp, n=3):
    q_vec = nlp(query).vector.reshape(1, -1)
    z_vecs = np.array([nlp(z).vector for z in zdania])
    sim = cosine_similarity(q_vec, z_vecs).flatten()
    top_i = np.argsort(sim)[::-1][:n]
    return [zdania[i] for i in top_i]

# Krok 4: Testy funkcji
print(znajdz_najblizsze("Jaki jest kurs dolara?", zdania, nlp_pl))
print(znajdz_najblizsze("Chce kupic nowy komputer", zdania, nlp_pl))
print(znajdz_najblizsze("Najlepsze restauracje w Krakowie", zdania, nlp_pl), '\n')

# Krok 5: Slowa wieloznaczne -- najblizsi sasiedzi
slowa = ["zamek", "klucz", "pilot"]
for s in slowa:
    t = nlp_pl.vocab[s]
    l = sorted([w for w in nlp_pl.vocab if w.has_vector and w.text != s], key=lambda w: t.similarity(w), reverse=True)
    print(f"Słowo: '{s}' -> Najbliżsi sąsiedzi: {[w.text for w in l[:8]]}")
print()

# Krok 6: Wizualizacja t-SNE wektorow zdan
groups = {
    "finanse": zdania[0:4],
    "technologia": zdania[4:8],
    "turystyka": zdania[8:12]
}

words, vectors, labels = [], [], []
for label, group in groups.items():
    for w in group:
        words.append(w)
        vectors.append(nlp_pl(w).vector)
        labels.append(label)

# t-SNE redukcja do 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(np.array(vectors))

# Rysowanie
fig, ax = plt.subplots(figsize=(10, 8))
colors_map = {"finanse": "red", "technologia": "blue", "turystyka": "green"}
plotted_labels = set()

for i, w in enumerate(words):
    group = labels[i]
    ax.scatter(coords[i, 0], coords[i, 1],
               c=colors_map[group], s=60,
               label=group if group not in plotted_labels else "")
    plotted_labels.add(group)
    ax.annotate(w, (coords[i, 0]+1, coords[i, 1]+1), fontsize=9)

ax.legend()
ax.set_title("Wektory zdan z korpusu -- wizualizacja t-SNE")
plt.tight_layout()
plt.show()

# Zadania cz. C

# Krok 1: Pobranie i preprocessing polskiej Wikipedii
ds = load_dataset("wikimedia/wikipedia", "20231101.pl", split="train[:1000]")
sentences_pl = []
for article in ds:
    text = article["text"]
    for line in text.split("\n"):
        tokens = simple_preprocess(line, deacc=False)
        if len(tokens) >= 3:
            sentences_pl.append(tokens)
print(f"\nLiczba zdan: {len(sentences_pl)}")

# Krok 2: Trening Skip-gram
model_sg = Word2Vec(
    sentences=sentences_pl,
    vector_size=100, window=5, min_count=5,
    sg=1, negative=10, epochs=10, workers=4, seed=42
)

# Krok 3: Trening CBOW
model_cbow = Word2Vec(
    sentences=sentences_pl,
    vector_size=100, window=5, min_count=5,
    sg=0, negative=10, epochs=10, workers=4, seed=42
)

# Krok 4: Porownanie Skip-gram vs CBOW
slowa = ["polska", "warszawa", "nauka"]

for s in slowa:
    print(f"\nNajblizsze do \"{s}\" wg. Skip-gram:")
    for word, score in model_sg.wv.most_similar(s, topn=5):
        print(f"    {word:15s} {score:.4f}")

for s in slowa:
    print(f"\nNajblizsze do \"{s}\" wg. CBOW:")
    for word, score in model_cbow.wv.most_similar(s, topn=5):
        print(f"    {word:15s} {score:.4f}")

# Krok 5: Analogie w wytrenowanym modelu
print(f"\npolska - warszawa + berlin = ")
for word, score in model_sg.wv.most_similar(positive=["polska", "berlin"], negative=["warszawa"], topn=5):
    print(f"    {word:15s} {score:.4f}")

print(f"\nkrol - mezczyzna + kobieta = ")
for word, score in model_sg.wv.most_similar(positive=["król", "kobieta"], negative=["mężczyzna"], topn=5):
    print(f"    {word:15s} {score:.4f}")

# Krok 6: Zapis i wczytanie modelu
model_sg.save("word2vec_sg.model")
model_loaded = Word2Vec.load("word2vec_sg.model")

print(f"\npolska - warszawa + berlin = ")
for word, score in model_loaded.wv.most_similar(positive=["polska", "berlin"], negative=["warszawa"], topn=5):
    print(f"    {word:15s} {score:.4f}")

print(f"\nkrol - mezczyzna + kobieta = ")
for word, score in model_loaded.wv.most_similar(positive=["król", "kobieta"], negative=["mężczyzna"], topn=5):
    print(f"    {word:15s} {score:.4f}")
# wyniki się nie różnią

# Krok 7: Wplyw parametru window
model_sg_2 = Word2Vec(
    sentences=sentences_pl,
    vector_size=100, window=2, min_count=5,
    sg=1, negative=10, epochs=10, workers=4, seed=42
)

model_sg_5 = Word2Vec(
    sentences=sentences_pl,
    vector_size=100, window=5, min_count=5,
    sg=1, negative=10, epochs=10, workers=4, seed=42
)

model_sg_10 = Word2Vec(
    sentences=sentences_pl,
    vector_size=100, window=10, min_count=5,
    sg=1, negative=10, epochs=10, workers=4, seed=42
)

print(f"\nNajblizsze do \"historia\" wg. Skip-gram (window=2):")
for word, score in model_sg_2.wv.most_similar("historia", topn=5):
    print(f"    {word:15s} {score:.4f}")

print(f"\nNajblizsze do \"historia\" wg. Skip-gram (window=5):")
for word, score in model_sg_5.wv.most_similar("historia", topn=5):
    print(f"    {word:15s} {score:.4f}")

print(f"\nNajblizsze do \"historia\" wg. Skip-gram (window=10):")
for word, score in model_sg_10.wv.most_similar("historia", topn=5):
    print(f"    {word:15s} {score:.4f}")
# wyniki różnią się
