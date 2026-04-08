from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def analyze_text(text):
    sentences = sent_tokenize(text)
    sentence_amount = len(sentences)
    token_amount = 0
    print(f"Ilość zdań: {sentence_amount}")
    for i in range(len(sentences)):
        tokens = word_tokenize(sentences[i])
        token_amount += len(tokens)
    print(f"Ilość tokenów: {token_amount}")
    print(f"Średnia ilość tokenów na zdanie: {token_amount / sentence_amount}")

text = (
    "Dr. Smith went to Washington. He arrived on Jan. 5th. "
    "The meeting was scheduled for 3 p.m. and lasted two hours. "
    "It was a productive day! Was it worth the trip? Absolutely."
)

analyze_text(text)
