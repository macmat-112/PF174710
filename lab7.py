import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

import re

print("Wszystkie biblioteki załadowane pomyślnie ✓")

# ============================================================
# Ćwiczenie A: Allegro Reviews — ładowanie datasetu
# ============================================================

from datasets import load_dataset
import pandas as pd

# Ładowanie datasetu z Hugging Face
dataset_allegro = load_dataset("allegro/klej-allegro-reviews")

print("Struktura datasetu Allegro Reviews:")
print(dataset_allegro)
print()

# Konwersja do DataFrame
df_allegro_train = pd.DataFrame(dataset_allegro['train'])
df_allegro_val = pd.DataFrame(dataset_allegro['validation'])
df_allegro_test = pd.DataFrame(dataset_allegro['test'])

print("Kolumny:", df_allegro_train.columns.tolist())
print(f"Zbiór treningowy: {len(df_allegro_train)} recenzji")
print(f"Zbiór walidacyjny: {len(df_allegro_val)} recenzji")
print(f"Zbiór testowy: {len(df_allegro_test)} recenzji")
print()

# Podgląd danych
print("Przykładowy rekord:")
print(df_allegro_train.iloc[0])
print()

# Rozkład ocen (rating: 1-5)
print("Rozkład ocen w zbiorze treningowym:")
print(df_allegro_train['rating'].value_counts().sort_index())

# Mapowanie etykiet tekstowych na czytelne polskie nazwy
label_names = {
    1.0: 'negatywny',
    2.0: 'negatywny',
    3.0: 'neutralny',
    4.0: 'pozytywny',
    5.0: 'pozytywny'
}

# Kolory przypisane do etykiet
label_colors = {
    'negatywny': '#e74c3c',
    'neutralny': '#95a5a6',
    'pozytywny': '#2ecc71'
}

# Dodajemy kolumnę z czytelną nazwą klasy
for df in [df_allegro_train, df_allegro_val, df_allegro_test]:
    df['label_name'] = df['rating'].map(label_names).fillna(df['rating'])

print("Mapowanie etykiet:")
for orig, name in sorted(set(zip(df_allegro_train['rating'], df_allegro_train['label_name']))):
    print(f"  '{orig}' → '{name}'")

# Wizualizacja rozkładu klas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, df) in zip(axes, [('Trening', df_allegro_train), ('Walidacja', df_allegro_val), ('Test', df_allegro_test)]):
    counts = df['label_name'].value_counts()
    # Sortujemy w stałej kolejności
    order = ['negatywny', 'neutralny', 'pozytywny']
    order = [o for o in order if o in counts.index]
    counts = counts.reindex(order)
    colors = [label_colors.get(o, '#999999') for o in order]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title(f'Zbiór: {name} (n={len(df)})')
    ax.set_ylabel('Liczba recenzji')
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('Rozkład klas sentymentu w Allegro Reviews', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Długość tekstów (w znakach i słowach)
df_allegro_train['char_len'] = df_allegro_train['text'].str.len()
df_allegro_train['word_count'] = df_allegro_train['text'].str.split().str.len()

print("\nStatystyki długości tekstów (treningowe):")
print(df_allegro_train[['char_len', 'word_count']].describe().round(1))

# Rozkład długości tekstów wg klasy sentymentu
fig, ax = plt.subplots(figsize=(10, 5))
for label_name in ['negatywny', 'neutralny', 'pozytywny']:
    subset = df_allegro_train[df_allegro_train['label_name'] == label_name]
    if len(subset) > 0:
        ax.hist(subset['word_count'], bins=50, alpha=0.5, label=label_name, color=label_colors.get(label_name, '#999999'))

ax.set_xlabel('Liczba słów w recenzji')
ax.set_ylabel('Częstość')
ax.set_title('Rozkład długości tekstów wg klasy sentymentu')
ax.legend()
ax.set_xlim(0, 500)
plt.tight_layout()
plt.show()

# Przykłady tekstów z każdej klasy
for label_name in ['negatywny', 'neutralny', 'pozytywny']:
    subset = df_allegro_train[df_allegro_train['label_name'] == label_name]
    if len(subset) > 0:
        example = subset['text'].iloc[0]
        print(f"--- {label_name.upper()} ---")
        print(example[:300] + "..." if len(example) > 300 else example)
        print()

def preprocess_text(text):
    """Podstawowy preprocessing tekstu polskiego."""
    # Zamiana na małe litery
    text = text.lower()
    # Usunięcie znaków specjalnych (zostawiamy polskie litery)
    text = re.sub(r'[^a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s]', ' ', text)
    # Usunięcie wielokrotnych spacji
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Zastosowanie preprocessingu
X_train = df_allegro_train['text'].apply(preprocess_text).values
y_train = df_allegro_train['label_name'].values

X_test = df_allegro_test['text'].apply(preprocess_text).values
y_test = df_allegro_test['label_name'].values

X_val = df_allegro_val['text'].apply(preprocess_text).values
y_val = df_allegro_val['label_name'].values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print()
print("Unikatowe etykiety:", np.unique(y_train))
print()
print("Przykład po preprocessingu:")
print(X_train[0][:200])

# Pipeline: TF-IDF + Naive Bayes
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),    # unigramy + bigramy
        min_df=2,
        sublinear_tf=True      # logarytmiczne skalowanie TF
    )),
    ('clf', MultinomialNB(alpha=1.0))
])

# Trening
pipeline_nb.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred_nb = pipeline_nb.predict(X_test)

# Raport klasyfikacji
print("=" * 60)
print("NAIVE BAYES — Raport klasyfikacji")
print("=" * 60)
print(classification_report(y_test, y_pred_nb))

# Pipeline: TF-IDF + LinearSVC
pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ('clf', LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'))
])

# Trening
pipeline_svm.fit(X_train, y_train)

# Predykcja
y_pred_svm = pipeline_svm.predict(X_test)

# Raport
print("=" * 60)
print("SVM (LinearSVC) — Raport klasyfikacji")
print("=" * 60)
print(classification_report(y_test, y_pred_svm))

# Pipeline: TF-IDF + Logistic Regression
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    ))
])

# Trening
pipeline_lr.fit(X_train, y_train)

# Predykcja
y_pred_lr = pipeline_lr.predict(X_test)

# Raport
print("=" * 60)
print("REGRESJA LOGISTYCZNA — Raport klasyfikacji")
print("=" * 60)
print(classification_report(y_test, y_pred_lr))

# Porównanie wyników
results = {
    'Naive Bayes': {
        'accuracy': accuracy_score(y_test, y_pred_nb),
        'f1_macro': f1_score(y_test, y_pred_nb, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_nb, average='weighted')
    },
    'SVM (LinearSVC)': {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'f1_macro': f1_score(y_test, y_pred_svm, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_svm, average='weighted')
    },
    'Regresja logistyczna': {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1_macro': f1_score(y_test, y_pred_lr, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_lr, average='weighted')
    }
}

results_df = pd.DataFrame(results).T
results_df.columns = ['Accuracy', 'F1 (macro)', 'F1 (weighted)']
print(results_df.round(4).to_string())

# Wizualizacja porównania
fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Porównanie klasyfikatorów na Allegro Reviews', fontsize=14, fontweight='bold')
ax.set_ylabel('Wartość metryki')
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.show()

# Macierze pomyłek dla wszystkich modeli
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_preds = [
    ('Naive Bayes', y_pred_nb),
    ('SVM (LinearSVC)', y_pred_svm),
    ('Regresja logistyczna', y_pred_lr)
]

# Ustalamy stałą kolejność etykiet
all_labels = sorted(np.unique(np.concatenate([y_test, y_pred_nb, y_pred_svm, y_pred_lr])))
display_labels = [label_names.get(l, l) for l in all_labels]

for ax, (name, y_pred) in zip(axes, models_preds):
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Macierze pomyłek — porównanie klasyfikatorów', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Analiza błędów najlepszego modelu (SVM)
y_pred_best = y_pred_svm

# Znajdź błędne predykcje
mask_errors = y_test != y_pred_best
errors_text = X_test[mask_errors]
errors_true = y_test[mask_errors]
errors_pred = y_pred_best[mask_errors]

error_df = pd.DataFrame({
    'tekst': errors_text,
    'prawdziwa_etykieta': [label_names.get(t, t) for t in errors_true],
    'predykcja': [label_names.get(p, p) for p in errors_pred]
})

print(f"Liczba błędnych predykcji: {len(error_df)} / {len(y_test)} "
      f"({len(error_df)/len(y_test)*100:.1f}%)")
print()

# Najczęstsze typy pomyłek
print("Najczęstsze typy pomyłek (prawdziwa → predykcja):")
confusion_pairs = error_df.groupby(['prawdziwa_etykieta', 'predykcja']).size()
confusion_pairs = confusion_pairs.sort_values(ascending=False)
print(confusion_pairs.head(10).to_string())

# Przykłady błędnych predykcji
print("Losowa próbka błędnych predykcji:")
print("=" * 80)
sample = error_df.sample(min(5, len(error_df)), random_state=42)
for _, row in sample.iterrows():
    print(f"PRAWDA: {row['prawdziwa_etykieta']} | PREDYKCJA: {row['predykcja']}")
    print(f"TEKST: {row['tekst'][:250]}...")
    print("-" * 80)

# Łączymy dane treningowe i walidacyjne do walidacji krzyżowej
X_cv = np.concatenate([X_train, X_val])
y_cv = np.concatenate([y_train, y_val])

print("Walidacja krzyżowa (5-fold, metryka: F1 macro)\n")

models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'SVM (LinearSVC)': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ('clf', LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'))
    ]),
    'Regresja logistyczna': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
    ])
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring='f1_macro', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:25s} → F1 macro: {scores.mean():.4f} ± {scores.std():.4f}")

# GridSearchCV na pipeline SVM
pipeline_grid = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2)),
    ('clf', LinearSVC(max_iter=10000, class_weight='balanced'))
])

param_grid = {
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    pipeline_grid,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_cv, y_cv)

print(f"\nNajlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy F1 macro (CV): {grid_search.best_score_:.4f}")

# Ewaluacja najlepszego modelu z GridSearch na zbiorze testowym
y_pred_best_gs = grid_search.best_estimator_.predict(X_test)

print("=" * 60)
print("NAJLEPSZY MODEL (GridSearch) — Raport klasyfikacji")
print("=" * 60)
print(classification_report(y_test, y_pred_best_gs))

# Najważniejsze cechy dla każdej klasy (na modelu SVM)
feature_names = pipeline_svm.named_steps['tfidf'].get_feature_names_out()
coefs = pipeline_svm.named_steps['clf'].coef_
classes = pipeline_svm.named_steps['clf'].classes_

n_top = 15
n_classes = len(classes)
n_cols = 2
n_rows = (n_classes + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for idx, class_label in enumerate(classes):
    ax = axes[idx]
    top_indices = np.argsort(coefs[idx])[-n_top:]
    top_features = feature_names[top_indices]
    top_weights = coefs[idx][top_indices]

    readable = label_names.get(class_label, class_label)
    ax.barh(range(n_top), top_weights, color=plt.cm.viridis(idx / max(n_classes - 1, 1)))
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_features)
    ax.set_title(f'Top {n_top} cech — klasa: {readable}', fontweight='bold')
    ax.set_xlabel('Waga cechy (współczynnik SVM)')

# Ukryj puste osie jeśli nieparzysta liczba klas
for idx in range(n_classes, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Najważniejsze cechy TF-IDF dla każdej klasy sentymentu',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Testowanie na własnych przykładach
nowe_teksty = [
    "Świetny lekarz, bardzo profesjonalny i kulturalny. Polecam gorąco!",
    "Hotel był brudny, jedzenie okropne, nigdy więcej.",
    "Wizyta u lekarza przebiegła bez problemów, nic szczególnego.",
    "Z jednej strony obsługa miła, ale pokoje wymagają remontu.",
    "Najgorszy hotel w jakim byłem. Karaluchy w łazience!",
    "Pani doktor poświęciła mi dużo czasu i dokładnie wszystko wyjaśniła."
]

# Preprocessing
nowe_teksty_processed = [preprocess_text(t) for t in nowe_teksty]

# Predykcja z najlepszym modelem (SVM)
predykcje = pipeline_svm.predict(nowe_teksty_processed)

print("Predykcje sentymentu na nowych tekstach:")
print("=" * 70)
for tekst, pred in zip(nowe_teksty, predykcje):
    readable = label_names.get(pred, pred)
    print(f"[{readable:>13s}]  {tekst}")

# Predykcja z prawdopodobieństwami (regresja logistyczna)
probas = pipeline_lr.predict_proba(nowe_teksty_processed)

print("Prawdopodobieństwa przynależności do klas (regresja logistyczna):")
print("=" * 80)
klasy = pipeline_lr.classes_
for tekst, proba in zip(nowe_teksty, probas):
    print(f"\nTekst: {tekst[:60]}...")
    for klasa, p in zip(klasy, proba):
        readable = label_names.get(klasa, klasa)
        bar = '█' * int(p * 30)
        print(f"  {readable:>13s}: {p:.3f} {bar}")
