from nltk.tokenize import word_tokenize

text = input("Podaj zdanie: ")
tokens = word_tokenize(text)
for i in range(len(tokens)):
    print(f"Token {i + 1}: {tokens[i]}")
