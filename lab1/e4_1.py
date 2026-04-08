from nltk.tokenize import sent_tokenize

text = "The experiment failed... Again. The results (see Fig. 2) were inconclusive. We need more data."
sentences = sent_tokenize(text)

print(f"Ilość zdań: {len(sentences)}")
print("Zdania:")
for i in range(len(sentences)):
    print(f"Zdanie {i + 1}: {sentences[i]}")

# tokenizator niepoprawnie rozbija zdanie drugie na dwie części
