from nltk.tokenize import word_tokenize

text = "I've been working at OpenAI since Jan. 2020, and it's been great!"

# tokeny: I; 've; been; working; at; OpenAI; since; Jan.; 2020; ,; and; it; 's; been; great; !;
# ilość: 16

tokens = word_tokenize(text)
print(f"tokeny: {tokens}\nilość: {len(tokens)}")
