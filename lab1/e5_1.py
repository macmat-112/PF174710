from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = "Warszawa jest stolica Polski. Liczy ok. 1,8 mln mieszkancow. W miescie znajduje sie wiele zabytkow, m.in. Zamek Krolewski i Lazienki."
sentences = sent_tokenize(text, language="polish")
for i in range(len(sentences)):
    tokens = word_tokenize(sentences[i], language="polish")
    print(f"Zdanie {i + 1}: {sentences[i]}")
    print(f"    Tokeny ({len(tokens)}): {tokens}")
