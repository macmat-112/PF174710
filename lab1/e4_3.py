from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = "They may have differentiated into the tribes Alamanni, Hermunduri, Marcomanni, Quadi, and Suebi by the first century AD. By that time the Suebi, Marcomanni, and Quadi had moved southwest into the area of modern-day Bavaria and Swabia. In 8 BC, the Marcomanni and Quadi drove the Boii out of Bohemia. The term Suebi is usually applied to all the groups who moved into this area, although later in history (around 200 AD) the term Alamanni (meaning \"all-men\") became more commonly applied to the group."
sentences = sent_tokenize(text)
for i in range(len(sentences)):
    tokens = word_tokenize(sentences[i])
    print(f"Zdanie {i + 1}: {sentences[i]}")
    print(f"    Tokeny ({len(tokens)}): {tokens}")
