import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import spacy

print("Wszystko gotowe!")


model = api.load("glove-wiki-gigaword-100")

# Rozmiar slownika i wymiar
print(f"Slownik: {len(model)} slow")
print(f"Wymiar wektora: {model.vector_size}")

# Wektor dla slowa
vec = model['python']
print(f"Ksztalt: {vec.shape}")
print(f"Pierwsze 10 wartosci: {vec[:10]}")

# Najbardziej podobne slowa
similar = model.most_similar("king", topn=10)
print("Najbardziej podobne do 'king':")
for word, score in similar:
    print(f"  {word:15s} {score:.4f}")

# Podobienstwo miedzy dwoma slowami
sim = model.similarity("cat", "dog")
print(f"\nPodobienstwo cat-dog: {sim:.4f}")

sim2 = model.similarity("cat", "car")
print(f"Podobienstwo cat-car: {sim2:.4f}")

# Ktory nie pasuje?
outlier = model.doesnt_match(["cat", "dog", "fish", "computer"])
print(f"\nNie pasuje: {outlier}")

# king - man + woman = ?
result = model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)
print("king - man + woman =")
for word, score in result:
    print(f"  {word:15s} {score:.4f}")

# Inne analogie
analogies = [
    (["paris", "poland"], ["france"], "stolice"),
    (["bigger", "small"], ["big"], "stopniowanie"),
    (["walking", "swam"], ["walked"], "czas gramatyczny"),
]
for pos, neg, label in analogies:
    res = model.most_similar(positive=pos, negative=neg, topn=1)
    print(f"\n{label}: {res[0][0]} ({res[0][1]:.4f})")

groups = {
    "animals": ["cat", "dog", "fish", "bird", "horse", "cow"],
    "colors": ["red", "blue", "green", "yellow", "black", "white"],
    "countries": ["france", "germany", "poland", "italy", "spain", "japan"],
}

words, vectors, labels = [], [], []
for label, group in groups.items():
    for w in group:
        words.append(w)
        vectors.append(model[w])
        labels.append(label)

# t-SNE redukcja do 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(np.array(vectors))

# Rysowanie
fig, ax = plt.subplots(figsize=(10, 8))
colors_map = {"animals": "red", "colors": "blue", "countries": "green"}
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

nlp = spacy.load("en_core_web_md")

# Wektor pojedynczego tokena
token = nlp("cat")[0]
print(f"Wektor 'cat': wymiar {token.vector.shape}")
print(f"Czy ma wektor: {token.has_vector}")

# Podobienstwo miedzy dokumentami
doc1 = nlp("I like dogs")
doc2 = nlp("I enjoy cats")
doc3 = nlp("The stock market crashed")
print(f"\ndogs vs cats: {doc1.similarity(doc2):.3f}")
print(f"dogs vs stock: {doc1.similarity(doc3):.3f}")

# Przykladowy korpus
corpus_text = """
Uczenie maszynowe to dziedzina sztucznej inteligencji.
Sieci neuronowe sa fundamentem glebokiego uczenia.
Przetwarzanie jezyka naturalnego wykorzystuje modele jezykowe.
Word embeddings reprezentuja slowa jako wektory liczbowe.
Klasyfikacja tekstu to jedno z podstawowych zadan NLP.
Analiza sentymentu pozwala okreslic wydzwiek opinii.
Transformery zrewolucjonizowaly przetwarzanie jezyka naturalnego.
Tokenizacja to pierwszy krok w przetwarzaniu tekstu.
Modele jezykowe przewiduja nastepne slowo w sekwencji.
Korpus to zbior tekstow uzywany do trenowania modeli.
"""

# Tokenizacja
sentences = [
    simple_preprocess(line)
    for line in corpus_text.strip().split("\n")
    if line.strip()
]

print(f"Liczba zdan: {len(sentences)}")
for s in sentences[:3]:
    print(s)

# Trening modelu Skip-gram
model_sg = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    negative=5,
    epochs=100,
    seed=42,
)

print(f"Slownik: {len(model_sg.wv)} slow")
print(f"Wymiar: {model_sg.wv.vector_size}")

# Najblizsze slowa
print("\nNajblizsze do 'jezyka':")
for word, score in model_sg.wv.most_similar("jezyka", topn=5):
    print(f"  {word:20s} {score:.4f}")
