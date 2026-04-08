from nltk.tokenize import word_tokenize

text = "Mrs. O'Brien can't believe it's 3:45 p.m. already!"
words = split(text)
tokens = word_tokenize(text)
for i in range(len(tokens)):
    print(f"Token {i + 1}: {tokens[i]}")
