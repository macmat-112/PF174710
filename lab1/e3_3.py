from nltk.tokenize import word_tokenize

text = "Mrs. O'Brien can't believe it's 3:45 p.m. already!"
words = text.split()
tokens = word_tokenize(text)

# słowa: Mrs.; O'Brien; can't; believe; it's; 3:45; p.m.; already!;
# ilość: 8

for i in range(len(words)):
    print(f"Słowo {i + 1}: {words[i]}")
print('\n')

# tokeny: Mrs.; O'Brien; can; 't; believe; it; 's; 3:45; p.m.; already; !;
# ilość: 11

for i in range(len(tokens)):
    print(f"Token {i + 1}: {tokens[i]}")

# poprawki: "[...] ca; n't; [...]" zamiast "[...] can; 't; [...]"
