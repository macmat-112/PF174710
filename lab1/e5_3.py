from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def count_unique_tokens(text, language="english"):
    sentences = sent_tokenize(text, language=language)
    unique_tokens = set()
    for i in range(len(sentences)):
        tokens = word_tokenize(sentences[i], language=language)
        unique_tokens.update(tokens)
    return len(unique_tokens)

text = "Something something, he said. He said it was a very important something, he said."
print(f"Ilość unikalnych tokenów w tekście: {count_unique_tokens(text)}")

# pary "Something" i "something", "He" i "he" są na chwilę obecną uznawane za dwa różne tokeny.
# jeżeli to potrzebne, można to zmienić, na przykład sprowadzając
# wszystkie tokeny do lowercase przed dodaniem do zbioru unikalnych wartości.

def count_unique_tokens_lower(text, language="english"):
    sentences = sent_tokenize(text, language=language)
    unique_tokens = set()
    for i in range(len(sentences)):
        tokens = word_tokenize(sentences[i], language=language)
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        unique_tokens.update(tokens)
    return len(unique_tokens)

print(f"Ilość unikalnych tokenów w tekście: {count_unique_tokens_lower(text)}")
