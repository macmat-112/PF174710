import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import spacy

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

# Krok 3: Podobienstwo cosinusowe miedzy parami
for p in [("doctor", "nurse"), ("doctor", "airplane"), ("happy", "sad"), ("happy", "joyful")]:
    sim = model.similarity(p[0], p[1])
    print(f"Podobienstwo {p[0]}-{p[1]}: {sim:.4f}")

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

print(f"Korpus: {len(zdania)} zdan")
for i, z in enumerate(zdania):
    print(f"  D{i:2d} [{kategorie_zdan[i]:>12s}]: {z}")
