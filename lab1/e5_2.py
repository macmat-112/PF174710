from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = "Warszawa jest stolica Polski. Liczy ok. 1,8 mln mieszkancow. W miescie znajduje sie wiele zabytkow, m.in. Zamek Krolewski i Lazienki."
sentences = sent_tokenize(text)
for i in range(len(sentences)):
    tokens = word_tokenize(sentences[i])
    print(f"Zdanie {i + 1}: {sentences[i]}")
    print(f"    Tokeny ({len(tokens)}): {tokens}")

# angielski tokenizator nie rozpoznał polskiego "m.in." i postanowił zakończyć na nim zdanie.
